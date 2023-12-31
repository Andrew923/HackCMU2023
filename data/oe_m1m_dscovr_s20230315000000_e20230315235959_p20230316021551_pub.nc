CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230315000000_e20230315235959_p20230316021551_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-16T02:15:51.099Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-15T00:00:00.000Z   time_coverage_end         2023-03-15T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxn)@  T          A Q��\)�W
=��z�B�CH���\)@XQ������nG�B�
=                                    Bxn)�  
(          @�{�Q��G����u�C6ٚ�Q�@}p���ff�\p�B��                                    Bxn)-�  T          @�p�����?n{���=C33����@�����C
=Bݮ                                    Bxn)<2  �          A녿�p�?&ff����RCp���p�@����ʏ\�M�B�Ǯ                                    Bxn)J�  �          A �Ϳ�\)?�����(���C� ��\)@�{����8�\Bۏ\                                    Bxn)Y~  T          A�ÿ�׾����	p���C@����@�  ��
=�hz�B㞸                                    Bxn)h$  �          A
=q��=q�G���=qCq���=q@/\)�#�B���                                    Bxn)v�  
�          A�R����z���Rz�C`����@J�H���=qB�(�                                    Bxn)�p  �          A\)���H��z���=qCeE���H@U���p��qB�                                    Bxn)�  "          A  ��33�(��Q�p�Cp쿳33@ff���u�B��                                    Bxn)��  T          A���.{�*�H���H=qC�'��.{?�� ��{B�G�                                    Bxn)�b  "          A�>���:=q����8RC��>��?�\)� Q��HB�                                    Bxn)�  
Z          A
=���ÿ����\��Cg�����@*=q��G��B��H                                    Bxn)ή  �          Ap��8Q�xQ����R  CF�f�8Q�@R�\��G��h�RC33                                    Bxn)�T  T          @��
���Ϳ�33��33G�Cb  ����@7
=��p�\)B�                                     Bxn)��  
�          @���녾k����
��C:^���@w���=q�a�B�k�                                    Bxn)��  T          A�\�QG�?�ff��z�33C"5��QG�@�=q��ff�1z�B�33                                    Bxn*	F  
�          A=q�<��@Q���� C:��<��@�=q��Q��ffB���                                    Bxn*�  T          A�\� ��?����HL�C%z�� ��@���˅�N��B��                                    Bxn*&�  �          @�p�����>��H�����C�H����@�����=q�T\)B��                                    Bxn*58  T          Aff��ff?��
���R� C�ÿ�ff@�33��{�F��BД{                                    Bxn*C�  
�          A
=�\)@����G�Cff�\)@�{�������B�aH                                    Bxn*R�  "          @��H��(�?��\����C��(�@��H����@  B�\)                                    Bxn*a*  "          A=q��@Dz���(��vffB�G���@�z�������B�                                    Bxn*o�  �          AG��N�R@����R�4z�B�33�N�R@�  �ff���
B�                                    Bxn*~v  T          AG��]p�@�����
=�4�\B�Ǯ�]p�@��
�����33B�k�                                    Bxn*�  
�          A�^{@x����{�G�\C�q�^{@ҏ\�K���G�B랸                                    Bxn*��  
�          A��L��@N�R�أ��`z�C��L��@ʏ\�������B�{                                    Bxn*�h  �          A z��8Q�@Y����ff�a�C8R�8Q�@�{�xQ���z�B�(�                                    Bxn*�  �          @�\)�L��?8Q��ָR��HC'L��L��@�p�����5=qB���                                    Bxn*Ǵ  
\          @�\�"�\�
=����C[�H�"�\?�ff�ۅ#�C�                                    Bxn*�Z  �          @���G��Tz���\CU����G�@C�
��
=�{��B�                                    Bxn*�   �          @�=q���
?�G���{�qCuÿ��
@��\�����3�
BՅ                                    Bxn*�  
�          @�zῸQ�@�{������B�Q쿸Q�@��Y����=qB�p�                                    Bxn+L  T          @�(���{@�������ffB��)��{@�(������AG�B̮                                    Bxn+�  T          @�G���33@�=q��\)�
=BӸR��33@陚�8Q�����B̀                                     Bxn+�  "          @�33��R@�p������
=B�8R��R@�
=�Ǯ�?\)B�G�                                    Bxn+.>  
�          @��N{@����u���p�B�p��N{@�
=��{�%�B�33                                    Bxn+<�  T          @����%@����Q��
=B�\�%@߮�\�:=qBܣ�                                    Bxn+K�  �          @����R@�p������B��B�=��R@�ff�\)��=qB�{                                    Bxn+Z0  
�          @陚����@   �����B�HC.����@�
=�QG��؏\C�                                    Bxn+h�  T          @ᙚ����@�
��
=�)33C�
����@�ff�333���
C=q                                    Bxn+w|  �          @�=q��{?��
��z��I�C���{@�z��q���CO\                                    Bxn+�"  
�          @�z���Q�@���z�� ��C޸��Q�@��
���H�o\)B�z�                                    Bxn+��  �          @�
=��
=@mp���33���C�{��
=@��R�
�H���RC�{                                    Bxn+�n  !          @������\@y�����R�C\���\@���޸R�RffCs3                                    Bxn+�  
�          @�����@�z��A���
=C�����@�G��\)����C�                                     Bxn+��  
�          @��Vff@���W���
=B�.�Vff@��H=�G�?W
=B��)                                    Bxn+�`  
�          @���\(�@�{���H�&�Cff�\(�@Å��z��tQ�B���                                    Bxn+�  T          @�R�^�R@�{��\)�#C�q�^�R@��������j{B��H                                    Bxn+�  
�          @���L��@�ff�����=qB���L��@�Q쿘Q����B�33                                    Bxn+�R  !          @��
�s33@|(���  �C��s33@��R��G��f�RB�33                                    Bxn,	�  U          @޸R�\(�@  ��ff�S�HC���\(�@����]p���B�u�                                    Bxn,�  S          @�{�j=q@mp����\�%�C���j=q@�녿�������B�                                    Bxn,'D  
�          @�Q��\(�?�=q���H�\\)C�3�\(�@���s33��\B��                                    Bxn,5�  �          @�{�W�@C33���HffC	���W�@�{�A�����B�z�                                    Bxn,D�  
�          @�
=�.�R@5����R�bp�C�q�.�R@����fff��Q�B�=                                    Bxn,S6  �          @�G��L��?�G���ff�i��C(��L��@��������RB��\                                    Bxn,a�  "          @�����p�?���{�J�C-���p�@S33��33�33CL�                                    Bxn,p�  S          @�z�����>\�����H(�C.�����@@����{�ffC�3                                    Bxn,(  
�          @����s�
��������X  CG�
�s�
?�  ��ff�Q�
CT{                                    Bxn,��  "          @�\)������z���G��4�\CE(�����?�ff��=q�6  C$�                                    Bxn,�t  S          @�G��X�ÿУ���\)�^33CM�H�X��?�(������`�HC�=                                    Bxn,�  	�          @���ff@z��ʏ\��B�Q쿦ff@�=q���\�\)B���                                    Bxn,��  T          @�=q���?��H���G�B��f���@��R���
�!�B�k�                                    Bxn,�f  �          @�33���?���=q.C�H���@�G������2��B�=                                    Bxn,�  T          @�p��HQ��G���ff�bp�CQc��HQ�?�=q���\�jffC��                                    Bxn,�  �          @����g���\�����g{C;�3�g�@%���
=�F(�Cz�                                    Bxn,�X  �          @�=q��Q�L����\)�^�C?T{��Q�@�H�����F�
C޸                                    Bxn-�  �          @�{�mp���G��ʏ\�nQ�C5���mp�@L�����R�@ffCE                                    Bxn-�  �          @��hQ�fff��G��m�CA�q�hQ�@\)���H�T{C}q                                    Bxn- J  "          @�ff�y����p����b�CEz��y��@
=q��p��T(�C
=                                    Bxn-.�  S          @���G����
��(��c33CBL���G�@�����O�C
=                                    Bxn-=�  U          @�R�HQ���ҏ\��C<^��HQ�@AG����
�U�HC\                                    Bxn-L<  �          @���S33�u�߮k�C4�R�S33@dz���Q��L  C                                    Bxn-Z�  S          @�p��r�\�Tz���(��rG�C@h��r�\@6ff��=q�Tz�C�                                    Bxn-i�  
)          @�33��������
=�[G�C:5����@3�
��=q�<�C0�                                    Bxn-x.  �          @�Q������
=���\�N��CC�\���?�\)��z��E(�C(�                                    Bxn-��  T          @����Q녿�G��ȣ��pG�CL���Q�?��������h��CQ�                                    Bxn-�z  
Z          @����ff?�
=��G��M��C@ ��ff@����}p��G�C
                                    Bxn-�   
�          @���z�?�{��G��R�
C"����z�@�����
=�\)C�                                     Bxn-��  �          @�z��dz�>����
=�w=qC,���dz�@w
=��Q��9p�C�                                     Bxn-�l  "          @���p  ?Y����Q��ez�C':��p  @tz����R�%Q�Cs3                                    Bxn-�  �          @������?Tz���{�"Q�C+L����@G��e��33C                                      Bxn-޸  T          @�����333����O�RC=W
���@�R��=q�:ffCO\                                    Bxn-�^  "          @陚�����H���R�<��CR=q���?�����R�V��C+��                                    Bxn-�  
�          @�\�s33�&ff��G��C�CV^��s33>��H���
�c�\C,�f                                    Bxn.
�  �          @����hQ���33�5���ffCj���hQ��:�H��G��<��CZ�{                                    Bxn.P  �          @�{���
�33����0�CO����
>��H����IffC-�{                                    Bxn.'�  �          @�=q���R�=q��{�A(�CQ�����R?8Q������X�\C*G�                                    Bxn.6�  "          @�Q��S�
�I������H�
C_���S�
=�G��ʏ\�yz�C2.                                    Bxn.EB  �          @���9���z�H��z��:=qCi���9���:�H��p��CB+�                                    Bxn.S�  �          @�\)��{�9����ff�o�CmY���{?\)��33B�C#G�                                    Bxn.b�  �          @��H��Q��c�
�����]G�Cw�3��Q�aG����H
=C<�R                                    Bxn.q4  �          @ۅ�.{�o\)��(��a�\C����.{��{��  ­��Cs�q                                    Bxn.�  T          @��H=L�������R�W=qC�]q=L�Ϳ.{��G�¨��C�=q                                    Bxn.��  �          @��Ϳ\)�o\)�����g��C�� �\)�\)���Hª�RCB                                    Bxn.�&  T          @�=q�����g
=�����i�
C��������������®�)CF�                                    Bxn.��  
Z          @�33�
=q�b�\����ip�C����
=q���
�ٙ�ª�RC<��                                    Bxn.�r  �          @أ׿�ff��33����;
=CzaH��ff���R��{�CZ�                                     Bxn.�  
Z          @�G�����=q�����7�HC��q�녿˅�أ��fCzE                                    Bxn.׾  �          @�  ��
=������5{C|\��
=������k�C`�{                                    Bxn.�d  
Z          @�
=�(�����������/�HC���(�ÿ�  ���=Cy=q                                    Bxn.�
  
�          @�  �(����(����4=qC�*=�(�ÿٙ���
=�Cx޸                                    Bxn/�  �          @�  �   ��p������@\)Ctff�   �z�H�љ�� CN&f                                    Bxn/V  T          @�Q��X���hQ����
�1�Cb��X�ÿ(����G��q  C>�3                                    Bxn/ �  
�          @����ff�w����
�L�Cn�ff���H�ٙ�(�C?�)                                    Bxn//�  �          @����333�mp���  �A�Ch��333�
=q������C>�R                                    Bxn/>H  
�          @�\�I���i�����<Q�Ce#��I������=q�|��C=p�                                    Bxn/L�  �          @��
�R�\�^{��  �>��Cb���R�\��33�����x�C9�q                                    Bxn/[�  T          @��
�j=q�K����R�<G�C]��j=q����33�l=qC5�{                                    Bxn/j:  �          @���s33�tz���{�$�Ca(��s33�p����\)�c�CA��                                    Bxn/x�  T          @����]p������-��C[�
���Ǯ��{�_\)C9\)                                    Bxn/��  �          @�z��]p��XQ�����JC`ff�]p�=L������|Q�C3(�                                    Bxn/�,  �          @�33�w
=�Mp���  �H33C[Ǯ�w
=>u�ٙ��qz�C0p�                                    Bxn/��  T          @���W��5�����KffC[���W�>�=q���H�s��C/\)                                    Bxn/�x  �          @�ff�y��@���@��\B  B�L��y��@��@ÅBT�CY�                                    Bxn/�  T          @����Q�@��\@�z�A�z�B�  ��Q�@!�@�  BS33C�                                    Bxn/��  
�          @���h��@���@s33A��B�Q��h��@Fff@�{BQQ�C��                                    Bxn/�j  �          @����l(�@�Q�@��B��B�(��l(�@�@��B^p�C\)                                    Bxn/�  �          @��H��33@�{@^{A��
C���33@"�\@�\)B8�HC�                                    Bxn/��  T          @�(��e@���@hQ�A�\)B��\�e@/\)@���BO�HC��                                    Bxn0\  �          @�����
@��R@C�
A�G�C �����
@L��@�(�B3G�C��                                    Bxn0  �          @�Q���
=@�\)@l��A�G�C����
=@{@�{B@Q�C{                                    Bxn0(�  �          @�=q��{@�Q�@:=qA�{C8R��{@Fff@���B%C�H                                    Bxn07N  T          @�����Q�@��@ffA���CW
��Q�@[�@�B=qC�R                                    Bxn0E�  �          @�����\@S�
@��RB\)C�����\?�@�\)BF�C-ٚ                                    Bxn0T�  �          @����(Q�@}p�@ÅBP\)B���(Q�>���@��B���C+\)                                    Bxn0c@  T          @�{��
=?��@�p�B?��C$�R��
=����@��HB<�CFp�                                    Bxn0q�  T          @���,(�@�{@A�A�ffB�\�,(�@Mp�@��\BOffC�H                                    Bxn0��  T          @�\)��
=@���@�ffB (�C����
=@G�@�z�BK(�C\                                    Bxn0�2  T          @��R��G�@�=q@333A��
B�p���G�@N�R@�=qB1��CW
                                    Bxn0��  T          @�ff����@n�R@x��A�ffC�����?��@�\)B%��C'0�                                    Bxn0�~  �          @��
��33@�=q?�Q�AR{C�q��33@H��@n�RA�G�C�3                                    Bxn0�$  �          @�z�����@c33?���A!�Cz�����@�@9��A���C�{                                    Bxn0��  �          @�����@�녿�����RC�R����@���?�33AJffC��                                    Bxn0�p  �          @��H��p�@Z�H�^�R��{C�q��p�@�{����)G�C)                                    Bxn0�  �          @�
=�{�?����\
=CJ=�{�@��
��\)��CaH                                    Bxn0��  �          @�p��ڏ\?��
��Q���C'Y��ڏ\?���\)�\)C$O\                                    Bxn1b  �          @�G����
@�=q���H��Cp����
@�z�?s33@�z�C                                    Bxn1  �          @��H�θR@vff?�AeG�C5��θR@=q@c�
A���C�\                                    Bxn1!�  �          A ����=q@���?�G�@�G�C8R��=q@U�@B�\A�p�C�                                    Bxn10T  �          @�\)��@�  ?^�R@���C����@Q�@6ffA��Cٚ                                    Bxn1>�  T          A   ��33@tz�@
=A�p�C���33@�@{�A��C!                                    Bxn1M�  �          A ���ə�@�(�?�z�A#�C�3�ə�@U@`  A���C
=                                    Bxn1\F  T          @�����@�  ?�z�AE��C
����@c33@x��A�G�C�q                                    Bxn1j�  T          @����@��H<�>�  CJ=���@�p�@!G�A���C5�                                    Bxn1y�  �          @�{��@�ff?��
@��
C#���@s�
@Tz�AɮC=q                                    Bxn1�8  T          @�33��@-p�@�\A��RCT{��?�\)@G
=A�
=C'��                                    Bxn1��  S          @��\����@��?���A'�Cu�����?�  @{A�z�C&
                                    Bxn1��  �          @����ff@z�H?.{@�\)C����ff@E@   A��C5�                                    Bxn1�*  �          @�p���33@��H?�@x��C5���33@Tz�@��A���CQ�                                    Bxn1��  T          @�p�����@�
=@/\)A�33B�.����@j�H@�p�B.=qC	��                                    Bxn1�v  �          @�  ����@�33@L��AڸRC�����@{@�G�B8��CB�                                    Bxn1�  �          @�\)�z�H@�ff@ffA�(�B�z��z�H@�Q�@�B#�
C�q                                    Bxn1��  �          A   �\��@�G�@�
A�
=B����\��@�=q@��B'z�B�#�                                    Bxn1�h  "          Ap����@��?�33A z�B������@��@���B�HC�                                    Bxn2  �          Az��x��@��@:=qA��B���x��@�p�@���B1��Cc�                                    Bxn2�  �          A���O\)@��H@uA���B�ff�O\)@z=q@�G�BP�RC��                                    Bxn2)Z  "          A���=p�@�(�@�Q�B��B�k��=p�@/\)@�
=BsQ�C	=q                                    Bxn28   �          A�@��@�
=@�BB�z��@��@6ff@�ffBpQ�C�                                    Bxn2F�  
�          A�
�=p�@�33@��RBG�B�R�=p�@/\)@�p�Br\)C	8R                                    Bxn2UL  �          @�33��@�G�@�\)B{B�����@P��@ӅBi�B�aH                                    Bxn2c�  �          @��{�@��������B�B��{�@ʏ\?=p�@�p�B��                                    Bxn2r�  �          @�p���33@�(���Q���C �H��33@�G����
�{B��                                    Bxn2�>  T          A����@~{����4z�C  ���@Ǯ�H����{B�Ǯ                                    Bxn2��  �          A����@�G������$ffC!H���@ə��&ff��z�B��3                                    Bxn2��  "          Ap���
=@`  ���4�C�3��
=@�G��Tz���
=B�W
                                    Bxn2�0  �          A�����@�����(����C�����@���%���C��                                    Bxn2��  �          A����@Z=q�����\)C�H��@����A�����C�                                     Bxn2�|  �          A(���=q@\(����H�{C!H��=q@�ff�5���G�CQ�                                    Bxn2�"  �          A���G�@���ff�2{CJ=��G�@�
=��=q��p�C
{                                    Bxn2��  �          A	p��ƸR@(����G��\)C��ƸR@��H�X�����\C�                                    Bxn2�n  �          AQ���  @����G
=��C.��  @�(��k�����C�                                    Bxn3  �          A����=q@�Q��.{��
=C	�H��=q@�p�=���?5C�                                     Bxn3�  "          AQ�����@�  �0  ���CQ�����@���.{��C
B�                                    Bxn3"`  �          A���@�\)��ff���Cs3��@��?���A5�C�=                                    Bxn31  �          AG����@����2�HCǮ���@�Q�?�=qA\)CG�                                    Bxn3?�  �          A��@��
��  ���HB�{�@ᙚ@`  A�  Bخ                                    Bxn3NR  �          A  �*�H@��þ�(��AG�B��
�*�H@��@R�\A�ffB�\)                                    Bxn3\�  �          A�H�@�G�?��@vffB�ff�@Ӆ@��A�33B��                                    Bxn3k�  �          Ap��A�@�ff?�R@���B����A�@�  @�p�A�p�B���                                    Bxn3zD  �          A���r�\@�\��G��B�\B�R�r�\@�@C�
A��B�8R                                    Bxn3��  �          A��Y��@�z�=u>�G�B�z��Y��@Ϯ@a�A͙�B�W
                                    Bxn3��  T          A
=���H@�>�G�@FffBϽq���H@�G�@��A�p�B�(�                                    Bxn3�6  �          A  ��ffA ��<�>aG�B��)��ff@�33@s33A�
=B̳3                                    Bxn3��  �          A����RA�=�?Y��B�Ǯ���R@�=q@x��A�
=BӀ                                     Bxn3Â  
�          Aff���RA(�>���@G�B�����R@���@��A�z�Bǳ3                                    Bxn3�(  �          Ap��Y��Az�>#�
?���B��R�Y��@�@�Q�A�\B�aH                                    Bxn3��  �          A녿��
Aff?:�H@�G�B�W
���
@��
@�ffB=qB�33                                    Bxn3�t  T          A��ٙ�@�{?�=q@�=qB�8R�ٙ�@���@��B
�B�G�                                    Bxn3�  T          A��@���?��RA=qB�Q��@ə�@�
=B��B�Ǯ                                    Bxn4�  
�          A��@  @��@{A�
=B� �@  @�{@�Q�B*\)B�{                                    Bxn4f  
�          A{�H��@�G�@��
B�B�  �H��@Fff@��B`(�Cff                                    Bxn4*  �          A�\���A?У�A333B�ff���@�@�ffBp�B�W
                                    Bxn48�  
�          A�H��=q@�Q�?��HA`Q�B��쿊=q@�
=@��\B%(�B�u�                                    Bxn4GX  
�          A=q�Q�@��?(�@�ffB�  �Q�@�z�@��A��RB��                                    Bxn4U�  T          @�z��~{@��H?Q�@��
B�p��~{@���@Z�HA�z�C�)                                    Bxn4d�  �          A�
�`  @��=L��>���B�  �`  @�ff@R�\A�Q�B��H                                    Bxn4sJ  �          A	�����@���?�z�Ap�C:�����@�\)@z�HA�Q�Ck�                                    Bxn4��  	�          A����=q@�Q�>�z�?�(�C���=q@�p�@H��A�z�C	
=                                    Bxn4��  
Z          A(���G�@�G����AG�C�
��G�@�G�@�RA�=qC.                                    Bxn4�<  
�          A���أ�@��<�>W
=C���أ�@��@#33A�{CT{                                    Bxn4��  
�          A  ��
=@\(�?
=@w�C����
=@2�\@ffA_�C ��                                    Bxn4��  
Z          A  ���>#�
@*�HA���C2���׿�=q@(�A�ffC;s3                                    Bxn4�.  
�          A
ff�\)?8Q�@\)A�Q�C.�R�\)�   @"�\A�\)C7u�                                    Bxn4��  
(          A{��ff?�G�@A�A��C*���ff���@Q�A�=qC5��                                    Bxn4�z  
�          @�(���{>aG�@HQ�A�{C2E��{���R@8��A��\C=�                                     Bxn4�   T          @��
����Q�@a�AׅC6�q����
=@>{A��CC�\                                    Bxn5�  "          @�p�����?O\)@w�A�(�C-{���Ϳ��@s�
A�=qC<�                                    Bxn5l  
(          A����p�@R�\@�  A�{C����p�?�  @���B �RC*�                                    Bxn5#  "          A������@fff@�G�B	33C33����?��
@���B/�
C*T{                                    Bxn51�  "          A���ə�@e@���B p�CW
�ə�?���@�B&\)C)�=                                    Bxn5@^  
�          A����@C33@��B ��C.���?8Q�@�=qBC-xR                                    Bxn5O  �          AG�����?�p�@�G�B�
C#s3���;�z�@�ffBffC6�                                    Bxn5]�  
(          A(��ۅ@z�@i��A�=qC!O\�ۅ>�
=@��B Q�C0��                                    Bxn5lP  
Z          A�\��Q�@(�@��HB�C!s3��Q�#�
@��B�C4\                                    Bxn5z�  "          A=q��
=?�\@�p�BJp�C n��
=����@���BO(�CB�
                                    Bxn5��  	�          AG��˅?�  @���B\)C&�˅�n{@��B(�C<J=                                    Bxn5�B  
�          A  ��{@E�@�\)B)��Cu���{>8Q�@�  BH=qC2)                                    Bxn5��  
�          Az����@C�
@���B#�
Cc����>�  @ʏ\BA�C1}q                                    Bxn5��  
�          A
{���@aG�@��\B�CxR���?B�\@�33B>z�C,z�                                    Bxn5�4  
�          A���  @#33@�(�B��C����  ��@�ffB2��C5+�                                    Bxn5��  �          A�
��z�@o\)@e�AӮC����z�?޸R@��
B\)C$(�                                    Bxn5�  �          A(���(�@��\@X��AʸRC8R��(�@��@�33BG�C�q                                    Bxn5�&  �          A(����@�p�@
=qA���C�)���@x��@�33B

=C
=                                    Bxn5��  
Z          A33��\)@��@`  A�(�C����\)@5@��\B'=qC��                                    Bxn6r            A\)����@�=q@�=qA��
C�����@.�R@�z�B9�C�H                                    Bxn6  �          A��R�Q�A�=q?�z�@�G�B�R�Q�Ao�AQ�A�(�B�                                    Bxn6*�  "          A��R���
A��@�(�A/�Bٙ����
A�{AJ�\B�B���                                    Bxn69d  	�          A�  ���@�{�?��7  C	
���AF�R��
=�̸RB�
=                                    Bxn6H
  T          A�Q��+�A�\)@��Ao
=B�p��+�Ay��Ah��B(�B�\                                    Bxn6V�  �          A�G���A�(�@�A�ffB�����AP��Amp�B.�B�                                      Bxn6eV  
�          A����{Ah(�A#�Bp�B�=q��{A(�AyG�Bj��B�u�                                    Bxn6s�  
Z          A�����A���@�z�A�p�B�=q��AD��A@��B�B�                                    Bxn6��  
�          A�  �N�\AI���{���HC�3�N�\AuG��*�H��\C�                                    Bxn6�H  h          A����5G�Ajff���
��p�B�k��5G�AxQ�?�{@���B�=q                                    Bxn6��  
Z          A��R�-Ah(���z��}G�B����-At��?޸R@��B�                                    Bxn6��  
�          A����6�RA<���z�H�V�HC{�6�RAE�?��@�=qC�{                                    Bxn6�:  
Z          A����
=AG
=�6�H�  B�Q��
=A�p����H�m��B��                                    Bxn6��  "          A�����A%�}��DffC�H��A��
�\)���B��)                                    Bxn6چ  "          A�\)��A�G���{�\��B�B���A�z�@<(�A	��B�\                                    Bxn6�,  r          A�{�+33Ad����z����
B����+33A��R��(����\B�B�                                    Bxn6��  
(          A�\)�K33A
=�.�R�p�C���K33A[33��������C�
                                    Bxn7x  	�          A��������33�R=q�j  CnB�������l���fC:B�                                    Bxn7  �          A������
�Dz���\)�CKJ=���
@��R�����{(�C�                                    Bxn7#�  T          A������@@  �q��f�RC!�f���A��Dz��.(�Cp�                                    Bxn72j  |          A�Q���A/
=��
���
B��3��AV�R�,����HB��                                    Bxn7A  �          A����G�A�33�\��z�B�=q��G�A��\@�  A�G�Bݏ\                                    Bxn7O�  �          A�p�����A�ff���R�O�B��f����A���@c33A.�HBݏ\                                    Bxn7^\            A�33��p�Ahz���(���33B����p�A�33�!G��33B޽q                                    Bxn7m  
�          A�{���HA[\)���� {B�����HA����<���33B�(�                                    Bxn7{�  �          A���У�A�{���=qB���У�A���@�{A�33B�p�                                    Bxn7�N  T          A�  ����A�=q�p��ڏ\B�=q����A�Q�@�Q�A���B�                                    Bxn7��  
Z          A�(��_�
AZff�q��/33C�3�_�
A_\)@�H@�\)C�                                    Bxn7��  	�          A�\)�)p�A?\)�(Q��z�C��)p�At�������T��B�\)                                    Bxn7�@  
n          A�{��A�����\)���HB��
��A��
>W
=?��B�                                    Bxn7��  "          A�p���ffA��333��33B�#���ffA�Q�@��
A���B�u�                                    Bxn7ӌ  
�          A������A��
��ff����B��f���A���@�Q�A��RB�Q�                                    Bxn7�2  
�          A���\)A�  �W��(�B���\)A�@��HAPz�B��                                    Bxn7��  
�          A���)p�A��
@ ��@���B����)p�A~ffA#\)A�
=B�W
                                    Bxn7�~  
�          A�=q���A�(����
��G�B�{���A��@���A�B�z�                                    Bxn8$  "          A�G����A��@�p�A*=qB����A�(�AC
=B�RB�33                                    Bxn8�  �          A�{�C33A���@O\)AQ�B�u��C33Ax(�A,��A�p�C .                                    Bxn8+p  �          A��
�D��A�(�?O\)@�B�8R�D��A���A��A�z�B���                                    Bxn8:  T          A�{�  A�p�@33@��
B�u��  A�33A(��A�33B�#�                                    Bxn8H�  "          A�(��9�A�G���=q���HB����9�A��H@�33A�{B�q                                    Bxn8Wb  T          A�G��1A��\�����*=qB���1A�G�@�ffA��B�                                    Bxn8f  �          A���!G�A�z������B���!G�A�=�G�>�=qB���                                    Bxn8t�  T          A����=qAmG��p  �(�HB�=��=qA����p���p�Bܨ�                                    Bxn8�T  �          A�
=���HA`���pQ��.B�����HA�=q��G����B��f                                    Bxn8��  �          A����{A������\��p�B�u���{A��?���@�Q�B�u�                                    Bxn8��  T          A�ff��  A���������\B�����  A�p�?��@\(�B�k�                                    Bxn8�F  �          A�(���Q�AEG���\)�Y=qB�8R��Q�A��
�=����Q�B�                                      Bxn8��  
�          A�=q��  @׮����qB�� ��  AmG���=q�9��B�Ǯ                                    Bxn8̒  �          A�\)>W
=@K���G�¢p�B��\>W
=A7\)���
�c
=B��                                    Bxn8�8  �          A�ff���Aa�F�H��B������A����\)�o�B�33                                    Bxn8��  @          A�p��)G�AK��V�R��C�R�)G�A���޸R���\B�\)                                    Bxn8��  |          A�ff�6{A@���\(����CW
�6{A�G������ffB���                                    Bxn9*  T          A����p�AH���k33�+��B��p�A������\)B�u�                                    Bxn9�  "          A�z��$Q�Ax  �8Q���Q�B�
=�$Q�A��H��(��.=qB��                                    Bxn9$v  �          A���)A�{�ָR��z�B�{�)A�ff?��@�BꞸ                                    Bxn93  �          A�z��33A��������G�B�{�33A���@33@���B�=q                                    Bxn9A�  �          A�{�33A�ff�B�\���
B�\�33A�p�@�{Ah(�B�aH                                    Bxn9Ph  T          A���
=A�=q����W
=B��H�
=A���@�(�A��B�{                                    Bxn9_  �          A�����A���A z�A���B�Q���Ax(�A{�B)ffB��                                    Bxn9m�  �          A��R��\)A��A�RA�(�Bٽq��\)Ai�A{�B1��B垸                                    Bxn9|Z  �          A��R���A��
@ٙ�A�p�B�\���A�p�Aj=qB#p�B�G�                                    Bxn9�   �          A�����  As�
��\)���Cٚ��  Aj�H@�
=A8��C	�f                                    Bxn9��  "          A�  �b{A�p��\)��(�C�3�b{As�
@���A��
C�{                                    Bxn9�L  "          A���=qAF�R���
�r�\C���=qA[���33�c�
CQ�                                    Bxn9��  "          A������Av{����p�C� ����Ap��@�  A#
=C	^�                                    Bxn9Ř  T          A��\�f{A�ff�A����C&f�f{A��\@�33A(  C��                                    Bxn9�>  
�          A���d��A�����33�#�
C
=�d��A�=q@UAp�C �=                                    Bxn9��  T          A�(��VffA��׿�{�8Q�B����VffA�Q�@���A}C�                                    Bxn9�  �          A�
=�jffAj{��
���C
=�jffA�{�
=��  C�                                     Bxn: 0  T          A��H�U�A��R��ff�o\)B����U�A��\@�A{
=B��                                    Bxn:�  �          A�p��+�A�
=�Fff��=qB�Ǯ�+�A��\@��A\��B�                                      Bxn:|  �          A����(z�A���G���ffB��f�(z�A�\)@޸RA��HB��
                                    Bxn:,"  T          A��A�=q�(���  B��A�@���A��HB���                                    Bxn::�  
�          A����b{A�ff@���AYG�C\�b{AT��A8��A�(�C��                                    Bxn:In  �          Aɮ����AdQ�A7�A�(�C&f����Az�A��B!33Cz�                                    Bxn:X  
�          A�ff���HA���AffA��HC�����HAG
=Ay�B�C�                                     Bxn:f�  �          A�Q��hQ�A�(�?Ǯ@��CO\�hQ�Ah��@�\)A��C�                                    Bxn:u`  �          A�(��[\)A�Q�@aG�A{B���[\)A�ffA/\)A�{C��                                    Bxn:�  T          A¸R�f�RA��
?k�@�B��f�RA�Q�A��A��HCn                                    Bxn:��  "          A�(��f�RA�p�@\(�AffB��H�f�RA�(�A+�A��C��                                    Bxn:�R  �          A��H�Z{A��R����RB����Z{A�z�@�z�Aa��B�#�                                    Bxn:��  �          A��.�\A�ff���F�RB�Q��.�\A�G�@H��@�(�B�k�                                    Bxn:��  T          A�G���RA��?�z�@��\B��H��RA��HAp�A�(�B�\                                    Bxn:�D  T          A����p�A�������.�RB�Ǯ��p�A�\)@��A�33B��                                    Bxn:��  "          A�p��  A�(������
B�8R�  A�=qA	G�A�ffB�z�                                    Bxn:�  "          A�Q��  A�ff@���A,��B��H�  A�
=AG\)B �B��                                    Bxn:�6  �          A��\��(�A�(�@���A}��B�Q���(�A��A^{Bp�B��                                    Bxn;�  
�          A��H���HA�z�A�RA��RB�����HAap�Az{B2�B�                                    Bxn;�  
Z          A������HA��A�A�=qB܏\���HAh��An=qB*ffB���                                    Bxn;%(  "          A����Q�A��H@���A��B��f��Q�AjffAg\)B%33B�B�                                    Bxn;3�  "          A����	G�A�=q@�33A_
=B��f�	G�A�\)AIp�B

=B��f                                    Bxn;Bt  "          A�  �33A���@R�\A
{B�Q��33A�z�A+\)A�B�{                                    Bxn;Q  T          A����RA��@:�H@�{B�{��RA���A!Aܣ�B��                                    Bxn;_�  �          A����
�RA������
�b{B� �
�RA�p�@ff@�{B�33                                    Bxn;nf  
�          A�(��Q�A�{���
��\)B���Q�A�G�@ٙ�A��\B�                                     Bxn;}  
�          A��H�M�A�=q@|(�A,(�C 5��M�AXz�A{A߅Cu�                                    Bxn;��  T          A���2{A��׽��;�=qB����2{A�=q@�z�A��HB��R                                    Bxn;�X  
(          A����G�A��&ff���B�k��G�A�=q@���A�B�8R                                    Bxn;��  �          A�Q���A�{������HB�=q��A��
@��HAe��B�8R                                    Bxn;��  �          A��
��  A����1�� (�B��f��  A�{@��
AV�\B֊=                                    Bxn;�J  �          A�����\)A���r�\�/�
Bԙ���\)A�(�@g�A'�BԊ=                                    Bxn;��  �          A�{�y��A����]p�B�=q�y��A��@��
A�
=B�u�                                    Bxn;�  "          A�=q�   A���U���B䞸�   A�33@hQ�A)��B�                                    Bxn;�<  T          A�\)� z�A�
=����[�B��f� z�A���@
=q@�ffB�                                    Bxn< �  �          A�G���p�A����Q��L��Bнq��p�A��@E�A\)B�W
                                    Bxn<�  �          A������A��H�\)���B�\����A���G��4z�B�L�                                    Bxn<.  
Z          A�����G�AV=q�tz��:
=B�����G�A�G����33B���                                    Bxn<,�  �          A���(�A}��p�����B�#��(�A��H�J�H�33B�u�                                    Bxn<;z  �          A�����
Ar{�(����(�B������
A�����
�=p�B�G�                                    Bxn<J   T          A�p���G�Amp��1p���B�Ǯ��G�A�G���\)�\��B��                                    Bxn<X�  �          A����G�A}p��\)��\B�W
��G�A����E��B�
=                                    Bxn<gl  
�          A�������A~�\�!���HBس3����A�=q�\���"ffB���                                    Bxn<v  "          A�G�����A�
=�G���Q�B�L�����A�����{B�L�                                    Bxn<��  T          A�  ��p�At(��Q���Q�B�
=��p�A�p��{��33B�                                      Bxn<�^  
�          A�ff�ffAj=q���H��=qB�{�ffAw�?5@G�B�Ǯ                                    Bxn<�  �          A�(����A�\)�u�8Q�B����Ap��@�{A��B�.                                    Bxn<��  T          A���Q�Aq��?c�
@7
=B�p��Q�A[33@��A�(�B��\                                    Bxn<�P  T          A����9AmG�<#�
<��
C ��9A\z�@�\)A�G�C)                                    Bxn<��  
�          A��R�F�RAq�@S�
A{Cz��F�RAL��A
{AΣ�C&f                                    Bxn<ܜ  �          A�ff�9G�A|��@��A=B�� �9G�AR�HA{A�(�CO\                                    Bxn<�B  �          A�p���
A��R@��AI�B�\)��
Adz�A%�A��B���                                    Bxn<��  �          A�G��(��A���@^{A"�HB���(��A\��A�RAޣ�B��q                                    Bxn=�  T          A�  �#�
A��@���AH��B�\�#�
AXQ�AffA�=qB�B�                                    Bxn=4  �          A�G��!�A���@��HA>ffB�{�!�A_
=AA�\B��3                                    Bxn=%�  �          A����A�p�@W
=A33B�����AlQ�A=qA�RB��q                                    Bxn=4�  
�          A����
A��?W
=@�B�B���
A�G�@��A��B�W
                                    Bxn=C&  �          A�G���HA��H?Tz�@
=B�aH��HA��R@�Q�A�\)B��                                    Bxn=Q�  T          A����{A�z��
=�ƸRB�G��{A�\)@�
=AHz�B�k�                                    Bxn=`r  T          A�G��  A��H�\(��)G�B��f�  A��@-p�A��B�                                    Bxn=o  
Z          A���p�A>=qA=qB Q�B����p�@�ffAP��B>ffC�q                                    Bxn=}�  T          A�����AX  Ap�A�p�B�#����A��A`��B@\)C0�                                    Bxn=�d  �          A�Q����AHz�A�HA��C \���A�A\Q�B<��Cc�                                    Bxn=�
  T          A�(��  AIG�A\)B{B����  A�\A]�BDffC�q                                    Bxn=��  T          A���   AM�A"�\B\)B�R�   A�Aa��BJffC�3                                    Bxn=�V  "          A�G����AP��ALQ�BQ�B�� ���@�33A���B]�C
\)                                    Bxn=��  "          A�(���{A=qA[33B@ffC8R��{@w�A�33BuQ�C                                      Bxn=բ  �          A�ff�"ff@�z�AW
=BCp�Cٚ�"ff?5Aj�\B\�C/��                                    Bxn=�H  �          A�{�&{?@  A[
=BS{C/޸�&{��AP��BE�RCI�                                    Bxn=��  �          A�G��>{@y��ALz�B6�C!�{�>{�fffAUG�B@�
C8O\                                    Bxn>�  �          A�{�=�?�Ab=qBHQ�C1@ �=����AV�RB:�HCH�=                                    Bxn>:  	�          A�Q��p(�@"�\A�RB\)C*h��p(����A"�RB=qC8W
                                    Bxn>�  
�          A����uA�@љ�A�p�CG��u@�\)A�A癚C!.                                    Bxn>-�  
�          A�z���=qAff@��A��RCY���=q@���A�RA�z�C�\                                    Bxn><,  �          A�  ��{A\)@j�HA'
=C@ ��{@��H@ҏ\A�  C=q                                    Bxn>J�  �          A�����\)A��@��@��C����\)@�{@���Ax��C.                                    Bxn>Yx  6          A�
=���
A���
=����C}q���
Aff@,��A=qC��                                    Bxn>h  	�          A�
=�[�
A���+��
=C�3�[�
A=q?=p�@!G�C��                                    Bxn>v�  h          A�
=�MA
=������\C0��MA �ÿ������\C�R                                    Bxn>�j  T          A�{�up�A;���z��A��C�)�up�AF�H>W
=?�RC��                                    Bxn>�  �          A�p��l��AC��.�R�Ck��l��AF�R?ٙ�@���C�q                                    Bxn>��  "          A��\�Xz�A>=q��33��  C
�3�Xz�AR�H������33C�                                     Bxn>�\  h          A�p��7�A3�
��\��\)C�)�7�A_���z�����Cc�                                    Bxn>�  
�          A����O�
A333�G����
C:��O�
AT���l���4  CO\                                    Bxn>Ψ  �          A�p��Q�A<z���\��
=C	���Q�A^{�e��*=qCG�                                    Bxn>�N  T          A�ff�7
=AHz��	��G�Cc��7
=Ak��p���6{B���                                    Bxn>��  �          A��@  AX����{���C�\�@  Amp���33�\(�C ��                                    Bxn>��  "          A�p����AT���"�\�\)B�33���A�
��
=�z�HB晚                                    Bxn?	@  �          A��AYp��{��B���A�G�����d��B�L�                                    Bxn?�  �          A�G���RAX�����H���RB�\)��RAt(��  ��G�B�\                                    Bxn?&�  
Z          A�=q�%��Ae������X��B���%��Ao\)?h��@6ffB�W
                                    Bxn?52  �          A����@(�Ar�H�-p���
=C \)�@(�As
=@(Q�@�  C T{                                    Bxn?C�  �          A�G��QG�Am���\(��\)C^��QG�Ad  @�  AF=qC��                                    Bxn?R~  �          A����((�A�33�p���0��B���((�A�?���@��RB��                                    Bxn?a$  �          A����@(�A�����
�h��B�Q��@(�Ay@���AD  B�#�                                    Bxn?o�  "          A��R�K\)As�����@  C�
�K\)Af�R@�p�Ae��Cff                                    Bxn?~p  "          A��\�ڏ\A}��� ����{B♚�ڏ\A���p���p�B�L�                                    Bxn?�  T          A����=qA~�\�����p�B�Q���=qA���9���
�RB���                                    Bxn?��  �          A�  ��\)Avff�
=���B�(���\)A�p��?\)���B��f                                    Bxn?�b  
*          A�����A�\)����33Bڔ{���A�=q�Fff�33B�ff                                    Bxn?�  
�          A�����\A����G���\)B����\A���vff�7
=B��)                                    Bxn?Ǯ  �          A��R���HA�Q���R��  B�L����HA��R�<���Q�B�u�                                    Bxn?�T  T          A����G�A����=q����Bܳ3��G�A��Ϳ�Q���B�aH                                    Bxn?��  �          A�ff��A�G�����_\)B�8R��A��
?�
=@�B���                                    Bxn?�  
Z          A��R�p�A��\��z���=qB��p�A�ff>\)>���B�.                                    Bxn@F  "          A�����A��������{B�p����A�ff�\)����B�{                                    Bxn@�  
�          A�p����A�ff��33�>�RB�����A�p�?��@�{B��                                    Bxn@�  T          A�p���HAxz��������RB�  ��HA�(��33���HB�
=                                    Bxn@.8  
Z          A����Au��	G����B�q��A�ff�A��{B�                                     Bxn@<�  
          A�=q�陚A{33��\)��  B��H�陚A���ff��=qB�ff                                    Bxn@K�  
�          A�����  A����^�R�"�RB�=q��  A�  @+�@�=qB��                                    Bxn@Z*  	�          A��R���A��\>�  ?333B�����A��@��HA���B➸                                    Bxn@h�  
�          A�ff�
{A����ff�C33B��
{A�z�@�=qAbffB�3                                    Bxn@wv  T          A�����A����^�R�%B��H���A�(�@�R@��
B��                                     Bxn@�  
�          A��H���
A���   ���
B�Ǯ���
A��R@L(�A�B�{                                    Bxn@��  �          A��H��z�A�(�@<��A�B�k���z�A�ffA(�A�p�B�=q                                    Bxn@�h  
�          A�{��Q�A��@�A���B��)��Q�A\��AL  B��B�#�                                    Bxn@�  �          A���A�z�@ڏ\A��\B�.��AZ�RA>�HB33B��                                    Bxn@��  �          A���(�A�33@�
=A�  B�#���(�A]�AFffB��B�aH                                    Bxn@�Z  
�          A����ffA��?�33@���B�Q���ffAmG�@ڏ\A�33B�                                    Bxn@�   T          A��
��AdQ������G�B�33��Aqp������
B�Q�                                    Bxn@�  	t          A�����A0���(���CB����AW
=��ff����B���                                    Bxn@�L  
P          A�p����AG��E���5Q�C G����AP���	����p�B�(�                                    BxnA	�  
Z          A����Q�A<(���ff��B�aH�Q�A[33�o\)�O\)B�=q                                    BxnA�  
�          A�(��Q�A\)�#�
��RC	���Q�A8��������{C��                                    BxnA'>  
�          A�=q���HA33�@Q��:��C�q���HA9��(����B�
=                                    BxnA5�  
Z          A���(�@���4(��0��C��(�A(z��G���  C��                                    BxnAD�  	�          A�z��׮A��8Q��4(�B�Ǯ�׮AH(���{��p�B��                                    BxnAS0  "          A�
=��Q�@�(��\Q��c33B�W
��Q�A2ff�,z��#��B�\                                    BxnAa�  

          A���.ffA+�
�4z��'
=Ck��.ffA1G�?@  @0  C��                                    BxnAp|  T          A�
=�+
=@�33�(��33C\)�+
=Ap���\���
C

                                    BxnA"  
�          A�z����@�  �F{�KffC}q���A��p��(�C�                                    BxnA��  
�          A�(��
=@�=q�0���+�
C�)�
=A�������HC33                                    BxnA�n  
�          A{����A����
��CE���A4����  �q��C33                                    BxnA�  "          A}G��W\)@��H���
���C�f�W\)@��R�33�33C+�                                    BxnA��  
�          A�=q�d��@�z��fff�PQ�C��d��@޸R��p���=qC�                                    BxnA�`  �          A�\)�ep�@�G��7��!�C��ep�@�=q��(��\CaH                                    BxnA�  
�          A����k�
@�����p��ƸRC��k�
@�G�>�  ?fffC��                                    BxnA�  
�          A�=q�s�@�z�녿�p�C:��s�@�  ?��H@�=qC��                                    BxnA�R  |          A���a�@���?��@ָRC)�a�@�G�@�
=Ar{CB�                                    BxnB�  T          A��,��A&=q@ÅA�p�C��,��@��RA��B��C�)                                    BxnB�  �          A����#�
A,Q�@�=qA��
C�\�#�
Ap�A{B
�\C�
                                    BxnB D  "          A���
A"=qAffA�\)C���
@�33A.ffB*z�C33                                    BxnB.�  
�          A�z��3\)@�(�A�RA�z�C��3\)@�ffA#\)B33C0�                                    BxnB=�  "          A�{�%�@ᙚA�\B��C���%�@hQ�A9��B:C ��                                    BxnBL6  "          A�\)�#
=@ҏ\A)��B$��C+��#
=@>�RAA�BC
=C#��                                    BxnBZ�  �          A�  �)p�@��HA/33B-��C���)p�?}p�A=G�B@33C.�                                    BxnBi�  "          A|���-G�@�G�A�HB
��C  �-G�@|(�A*�\B+
=C �                                    BxnBx(  
(          Au��'�A-p�@
=A�\C��'�Ap�@��\A�C	��                                    BxnB��  T          A|Q��%p�A;�?�{@���CaH�%p�A)p�@��
A���CL�                                    BxnB�t  
�          Az�\�'�A3�@>�RA/�C��'�AQ�@���A�G�C	                                      BxnB�  "          At  �=qA-p�@�{A�{C�f�=qA\)@�G�A�z�C	
                                    BxnB��  T          A~�H�z�A?��J�H�9p�C:��z�AE�?#�
@z�C Q�                                    BxnB�f  "          A�(���\AS�
�s33�[�B�L���\A\Q�>�
=?�G�B�Q�                                    BxnB�  T          Ax  ���A@  �   ��33B������A9G�@K�AC\)B��                                    BxnB޲  
�          Az�\�:ffA�@�33A���C�H�:ff@�=q@�A��HC޸                                    BxnB�X  
Z          Ay����AEp�@A
ffB�����A0��@�\)A�Q�C��                                    BxnB��  
�          AxQ����AJ=q@G�A:�\B�����A1��@ٙ�A���B��                                    BxnC
�  �          Aw���HA4��@C33A9G�C�{��HA@�=qA���C}q                                    BxnCJ  �          Az=q�!�A4��@p��A_
=C���!�A=q@߮A��CB�                                    BxnC'�  T          A~{�&{A-@�p�A�(�C���&{A33@�\)A�33C=q                                    BxnC6�  	�          A}��33A2ff@Mp�AC�
C��33A�R@��Aȣ�C�                                    BxnCE<  
�          Ap���
=qA
=@�
=A�\)CxR�
=q@�z�AQ�B�Cp�                                    BxnCS�  
�          Aw��\)@��A#�B&p�C8R�\)@��A@  BM  C�                                     BxnCb�  "          A}����@��\A3
=B3�
C(���?�p�AE��BL�HC(�                                    BxnCq.  
�          A���Q�@��AAp�B@�C���Q�?J=qAN�RBS=qC/^�                                    BxnC�  
Z          A�ff��@~�RAD(�BC�CG���>\)AN=qBQ  C333                                    BxnC�z  �          A}G��!G�@��A1�B1�
CY��!G�?��AAG�BG=qC+W
                                    BxnC�   
�          Ay�=q@��HA5��B:�C���=q?��\AC33BN{C-�q                                    BxnC��  �          Ay��"{@��RA)��B+�RC��"{?�A:�RBBQ�C)��                                    BxnC�l  
�          Ax���2{@��Ap�B(�C��2{@=p�A&�HB(��C%�                                    BxnC�  
          Ayp��!�AG�A��B	
=C@ �!�@��
A,z�B.�RC
=                                    BxnC׸  "          A}���%�AG�@�RA��HC	#��%�@��A\)B�C                                    BxnC�^  
Z          A��� ��A5�@���A��
Cn� ��A  A	�B�C	\)                                    BxnC�  T          A}��(z�A(z�@��
A�z�C�q�(z�A�
A�A�ffC�                                    BxnD�  
�          A~�R�,Q�A*�\@��RA�(�CJ=�,Q�A��@�(�A�  C�R                                    BxnDP  "          A~ff�&{A0��@�\)A���C=q�&{A�R@�Q�A�  C
��                                    BxnD �  T          A�Q��+\)A-�@��RA�=qC���+\)Aff@��A��CG�                                    BxnD/�  �          A{���A,(�@��A���C�
��A
=qA��BC)                                    BxnD>B  
(          Av�H��\A'33@�=qA��C}q��\A�A�BQ�C��                                    BxnDL�  
�          Atz��!p�A&�H@�  A��C\�!p�A	p�@�\A�(�C�{                                    BxnD[�  "          Av�\��RA+�@���A���CǮ��RA�@�A�C
0�                                    BxnDj4  �          A���EG�A&�R@FffA-C���EG�A�@��A���C��                                    BxnDx�  T          A��Mp�A��?�\)@~{C��Mp�A��@z=qA_�C�
                                    BxnD��  �          A�  �F�HA%?���@��C0��F�HA��@�Q�As\)C�                                     BxnD�&  T          A��\�=G�A*ff@�
@�C
  �=G�A@�G�A�z�C�                                    BxnD��  
�          A�  �EG�A �þ�33���\CǮ�EG�A�
@#�
A(�C�3                                    BxnD�r  
<          A�G��I��A!�������=qCY��I��A�
@'
=A��CL�                                    BxnD�  �          A��H�?�
A,(�?��
@j�HC
��?�
A (�@��\Aj{C&f                                    BxnDо  �          Av{���HA<��@�
=A��RB�G����HAG�Ap�B(�CL�                                    BxnD�d  �          Ay��ffAE�@�p�A�
=B��q�ffA'
=A(�A��B��f                                    BxnD�
  �          As�
�{AE@<(�A2ffB�=q�{A0(�@�33A�\)B��=                                    BxnD��  �          Ak�����AHQ�@N�RAK33B����A1�@�p�A׮B���                                    BxnEV  �          Al�����AO33@7�A3
=B�q���A9p�@�{A�ffB�                                      BxnE�  "          Aj�\��
=AF�\@7�A4��B�\)��
=A1G�@���A�G�B�(�                                    BxnE(�  T          Au�� ��A*=q@w
=AlQ�Cc�� ��A@�
=A��HC	�
                                    BxnE7H  �          Aj=q�(Q�A�H@fffAd��C
#��(Q�A Q�@��
AƸRC�                                    BxnEE�  
�          Ajff�0z�A��?=p�@<(�Cٚ�0z�A�@UAV{C��                                    BxnET�  h          Ac��((�A��O\)�S�C�=�((�AQ�z��C	�
                                    BxnEc:  T          Ab�H�%�A{�P���U�C
z��%�A�H����\)C��                                    BxnEq�  T          Ae��#
=A  �U��V�HC	��#
=A �Ϳ��z�Cc�                                    BxnE��  
�          Af=q�+33AQ��Vff�W�Cٚ�+33A���.{�-p�C
{                                    BxnE�,  T          Ae�%�A���<(��<��C	=q�%�A��\)��C�                                    BxnE��  h          AhQ��#33A�\�0  �.�HC�{�#33A$z�>\)?\)C�                                    BxnE�x            Al���(  A��9���5G�Cu��(  A&=q<#�
<�CL�                                    BxnE�  �          Ah���@(�@��@  �>�\C�f�@(�A=q�@  �>�RC�)                                    BxnE��  T          Ab=q�M�@����"�\�%C�3�M�@��ÿxQ��{�C��                                    BxnE�j  T          AdQ��S�@�?0��@333C�)�S�@���@
=Az�C �                                    BxnE�  T          Aa��E�@�@*�HA/�CY��E�@�ff@�G�A�=qC!H                                    BxnE��  �          A\���@��@���@x��A��HCJ=�@��@�  @���A���C �
                                    BxnF\  
�          Ab{�D(�@�{@��A�p�C��D(�@Vff@�\)A���C$�q                                    BxnF  T          Ah(��<  @�\)@��A�33C��<  @��@��HA�(�C ��                                    BxnF!�  "          Ap(��=�@���@љ�AЏ\C�=�@�
=A z�BffC T{                                    BxnF0N            Aj�R�5��@�33@�(�A�Q�CǮ�5��@A�A��BffC%
=                                    BxnF>�  |          AaG��'\)@�@�  A��C�=�'\)@�(�@��A�C�                                    BxnFM�  �          AZ�\�{A��?��@��C�3�{A=q@n{A}�C��                                    BxnF\@  �          AXQ��A  @(�A
=C�)�A ��@��\A�z�C                                    BxnFj�  T          A^�\��A0  ��R�#�
B��=��A,  @�HA!p�B��
                                    BxnFy�  �          A^�\��\)A1���G�B�\)��\)A5p�?B�\@I��B�G�                                    BxnF�2  �          AZ=q� ��A*{�.{�8Q�B�33� ��A/�>�z�?��RB��                                     BxnF��  J          AX  ��ffA/33�C�
�R�\B�����ffA5�=#�
>B�\B��                                    BxnF�~  
�          AU����Ap��������B�3����A5��w�����B�{                                    BxnF�$  �          AP������A  �����
B�������A$����ff��  B�Ǯ                                    BxnF��  6          ALQ����AG���{��B噚���A0  ���
���RB���                                    BxnF�p  T          AR{��z�A
=������B�\��z�A0����������B�\                                    BxnF�  T          AQ���z�A���p��ffB�aH��z�A4����Q�����B�Ǯ                                    BxnF�  �          AP(�����A!����(�� �
B������A9p��x����=qB�ff                                    BxnF�b  �          AN�R��A����((�B�uÿ�A0Q����H����Bƞ�                                    BxnG  
�          AO�
���\A���p�� ��B�����\A3�
��G���
=B䞸                                    BxnG�  "          AQ����A
�R����*�RB�q���A+�
����=qB��                                    BxnG)T  
�          AU���33A����*�\B�����33A-��ə�����B�Ǯ                                    BxnG7�  T          AV�H��\)A{�����
=B�33��\)A.�R������B�\)                                    BxnGF�  �          AS\)��\)A\)��G��Q�B�����\)A-G��������B���                                    BxnGUF  T          AU���	A�R��\)���RC.�	A���\)�߮C��                                    BxnGc�  �          AS
=��(�A����Q���33C���(�A&{�Vff�l  B��)                                    BxnGr�  T          AR=q��(�A���z���33B�����(�A&�R�\)���
B��H                                    BxnG�8  J          AR=q��p�A�H����B�L���p�A)����R��=qB��\                                    BxnG��  �          AU�3�
A%��ff���B�L��3�
A@�������33B�8R                                    BxnG��  "          ATQ���\A�c�
�}�C+���\A�
�������RC5�                                    BxnG�*            AT����A$����(����B����A4�������B�G�                                    BxnG��  �          AV�\��G�A$�������z�B�����G�A333��=q����B�G�                                    BxnG�v  T          AVff�
�RA�\���r=qC�\�
�RA
=�Y���j=qC
                                    BxnG�  
�          ARff�A�
?��H@�z�CB��@���@i��A��C��                                    BxnG��  h          AW��"ffA
=q?�=q@��C�)�"ff@�33@���A�{CL�                                    BxnG�h  W          AV{�p�A�H?��R@�z�C	��p�A�@s33A��C�                                    BxnH  =          AV{�	�A
=�h���~=qCc��	�A �ÿ��\����C��                                    BxnH�  |          ATz��
=A��Z=q�o
=C�{�
=A%녿333�B�\C J=                                    BxnH"Z  
Z          AU��\AG��)���7
=C  ��\A\)�#�
�333C�
                                    BxnH1   
�          AR�H���A�H�:�H�N�\C�H���A!��\)����Ch�                                    BxnH?�  �          AV�H�{@�=q�߮��p�C���{@�
=��33����C��                                    BxnHNL  
�          AR�\��@�  ���R���
C.��A��^�R�up�C�q                                    BxnH\�  T          AO\)��G�A����Q��\)B��H��G�A$Q���
=��  B��f                                    BxnHk�  
�          ALz��Y��A!���(��(�B���Y��A9��\)���\BԽq                                    BxnHz>  
�          AN=q���
A.�\�����\)B�\���
A<�׿�����B��)                                    BxnH��  �          ALz��N�RA5p���33���HB����N�RAC���(���Q�Bѣ�                                    BxnH��  
�          AM����A#�������
B�.��A4  ����2{B�8R                                    BxnH�0  
�          AN=q�N�RA=���J�H�g\)B҅�N�RADQ�<�=�Bъ=                                    BxnH��  6          AL(���G�A����
=��\)B�{��G�A*=q��G����B��                                    BxnH�|  �          AB=q���\@��ff�0�B��H���\A��  ��B��)                                    BxnH�"  �          AB�\��{@�������B���{A  ��=q��\)B���                                    BxnH��  T          AC���G�A ����=q�=qB�p���G�A�����R�ԏ\B�\                                    BxnH�n  �          AE��\)@�����
=��
C�f��\)A��������(�C�                                    BxnH�  T          AB�H���A��޸R�ffB�{���A�
�������
B��                                    BxnI�  �          AE�����
Aff��p���RB��{���
AQ���������B���                                    BxnI`  6          AJ=q���A���.�R�N�HC{���A  ��z`\)C �H                                    BxnI*  J          APz���Aff?Tz�@i��C33��A
{@J=qA`��C�H                                    BxnI8�  �          AO��A�R?��A   C��Aff@��HA�(�C�=                                    BxnIGR  |          ALQ��
ffA��@�A���C#��
ff@޸R@�p�A��C.                                    BxnIU�  T          AK
=��A(�@S�
Ar�HCk���@��@�Q�A���CW
                                    BxnId�  �          AHz����AQ�?��A33CǮ���A�
@�(�A�G�CaH                                    BxnIsD  �          AG\)��\)AG�?�33Az�C�\��\)Az�@�
=A���C�                                    BxnI��  T          AH���  A�@��A��C���  A�@�p�A��C��                                    BxnI��  
�          AFff� ��A�H?Ǯ@���CJ=� ��A�@w
=A�  C�\                                    BxnI�6  T          AIG���
=A$(�?   @�B�G���
=A��@FffAd��B���                                    BxnI��  �          AF�R�ָRA%p��s33���B����ָRA#�
?�(�@��B�z�                                    BxnI��  �          AC\)��G�A,(���������B�z���G�A,��?�(�@�  B�G�                                    BxnI�(  
�          AD���ƸRA(�Ϳ�z���{B����ƸRA(  ?Ǯ@�G�B�33                                    BxnI��  
]          AF�R��p�A&�R�������B�=q��p�A%��?�\)@�Q�B���                                    BxnI�t  Q          AG���{A*ff�W
=�w�B�\)��{A((�?��A�B�                                    BxnI�  @          AD(���ffA   ����!�B�����ffA#\)>��@p�B��                                    BxnJ�  
�          A>�H��A{��� z�C\)��A  ?5@X��C ��                                    BxnJf  �          A>ff��
=A	��Vff����C���
=A
=�����33C�                                    BxnJ#  �          A@����33Ap���  ����CQ���33Aff�   �z�C��                                    BxnJ1�  �          A>�\��=qAG��������HC+���=qA��<(��dQ�B��\                                    BxnJ@X  
�          ADz��ᙚ@ʏ\���\�p�C
{�ᙚA����\)��
=C                                    BxnJN�  
�          A@(�����Az���\)��B�#�����A{�E�o
=B���                                    BxnJ]�  T          A>�H���Aff�������HB�33���A33�8Q��_�B�p�                                    BxnJlJ  T          A9G�����A33����p�B��q����A���G��z=qB�ff                                    BxnJz�  T          A8Q�����A(����H�ܸRB������AG��AG��s33B�                                    BxnJ��  T          A8(���Q�A���z���
=B��R��Q�AQ��W
=���B��                                    BxnJ�<  "          A5��{A������33B�aH��{A�\�J�H���B�\                                    BxnJ��  �          A5G��p  @���z��z�B����p  A�R����̣�B�p�                                    BxnJ��  �          A7�
�$z�@�{��\�;�B�.�$z�A���ȣ��  B��f                                    BxnJ�.  �          A:=q��
=@�����
�Q�B�W
��
=A{��p����B��H                                    BxnJ��  S          A+33��(�@�R�Mp���ffC���(�@��ÿ�ff���C}q                                    BxnJ�z  �          Az����@�ff�k���G�C(����@��Ϳ���2ffCO\                                    BxnJ�   �          A.�\��p�@���i������Cp���p�@���ٙ��(�C�f                                    BxnJ��  �          A5G���  @����z�H��  C��  A  ������HCs3                                    BxnKl  
�          A+\)����@�(��QG���{C�
����@�
=������\Ck�                                    BxnK  
�          A(Q���G�@�33�<�����C��G�@������  C�                                    BxnK*�  �          A*�\��@�Q�����P��Ch���@�(���\�0��C�q                                    BxnK9^  �          A-�� ��@ᙚ������{C
��� ��@��H?8Q�@w
=C
��                                    BxnKH  �          A'
=� ��@�G�>��?W
=C�f� ��@ə�?��
A��C��                                    BxnKV�  �          A �����@�\���R��ffC����@�{?��HA�C��                                    BxnKeP  T          A"{����@���������C������@�ff?0��@z=qCQ�                                    BxnKs�  "          A1���  @����{��=qC���  @��\?(�@H��C�3                                    BxnK��  
�          A8��� ��@��\�=q�Ap�C�\� ��A�H�Ǯ��
=C��                                    BxnK�B  �          A1���p�@���a���=qC�)��p�A�\�Ǯ��\CG�                                    BxnK��  �          A0Q����@���@  �~�RC!H���A�\�p����ffCh�                                    BxnK��  �          A-���(�Aff�G
=���C \��(�A
�R��  ��G�B��q                                    BxnK�4  �          A&�R��{@�R��p��z�B����{A=q�mp�����B�u�                                    BxnK��  �          A�\�c�
@�p���{�4�B����c�
@�R��Q����B�                                      BxnKڀ  �          A�y��@�z��ۅ�>z�C �
�y��@�{��{�\)B�z�                                    BxnK�&  T          A�����R@��\��Q��Q�C����R@�ff�i�����C (�                                    BxnK��  �          AQ���Q�@�G��C�
����C����Q�@�33��z��z�CB�                                    BxnLr  T          A
{��Q�@����|��C�
��Q�@љ��@  ���C)                                    BxnL  	�          A=q���@�>���?�(�B������@�{@Q�Ae��B�
=                                    BxnL#�  
�          Az�����@���<�>aG�B��H����@ᙚ?�=qA;�B���                                    BxnL2d  �          A�����@�=q@q�A���Cٚ����@�
=@�\)B�C�                                    BxnLA
  T          A���\)@�G�@�G�A�{C����\)@i��@��HB��Cc�                                    BxnLO�  T          Aff��=q@��
?fff@��C}q��=q@�\)@  A`��Cz�                                    BxnL^V  �          A�
���@��׿�  ��\)C� ���@�33>Ǯ@��CaH                                    BxnLl�  �          A  ��(�@љ�?��RA33C���(�@�\)@C�
A�33C�3                                    BxnL{�  "          A�����@�33@s33AŅC\)���@�\)@�B
��C
(�                                    BxnL�H  
�          A���z�@���@�=qA��
B�z���z�@�(�@�=qB(�C��                                    BxnL��  �          A�\���R@�{@A�A��C 5����R@�\)@�z�A�ffCff                                    BxnL��  
�          A����@��@���A�33C^����@���@�\)B�Cs3                                    BxnL�:  T          A���R@�Q�@�RA��RCǮ���R@�
=@xQ�AиRC
��                                    BxnL��  
�          Ap���G�@�=q?���AB�HC����G�@�@Q�A��C
                                      BxnLӆ  �          A\)���H@���?�{A1G�C� ���H@�
=@C33A�=qC�H                                    BxnL�,  �          AQ��ə�@�ff?��RAp�Cs3�ə�@���@>{A��\C
!H                                    BxnL��  "          A���*=q@a�@���BH��B���*=q@�@�  Bq��Cp�                                    BxnL�x  �          A=q��(�@Dz�@�\B���B�33��(�?�\)A��B�B�C��                                    BxnM  P          A���z�@
=q@��RB���C�
�z�>L��A (�B��HC.c�                                    BxnM�  
�          Ap��e�@��@�Q�BL�C^��e�@!G�@��Bs��C�f                                    BxnM+j  T          A=q��(�@�@�ffBQ�B�  ��(�@�@�BB�C��                                    BxnM:  �          A���w�@���@�(�B9=qCW
�w�@1G�@��B_�Cn                                    BxnMH�  
Z          @�Q��c33@;�@�ffBLffC}q�c33?���@ǮBi�
C�{                                    BxnMW\  
�          @�ff��?���@��
B9  C ٚ��>�\)@��HBDffC0.                                    BxnMf  T          @����  @Fff@��B/Q�C:���  ?���@�
=BL�
C�=                                    BxnMt�  T          @�
=�O\)@AG�@��B9��C�q�O\)?��@�Q�B[\)C�                                    BxnM�N  "          @����(�@5@�{B8
=C����(�?�p�@�
=BR�C 5�                                    BxnM��  "          AQ��\@p�@��B(�C   �\?fff@�  B'(�C+�{                                    BxnM��  �          A������@,(�@�B\Q�CE����?k�@�(�Bs{C'.                                    BxnM�@  �          A	���@�z�@��B33Cٚ��@@  @�=qB>�CY�                                    BxnM��  �          A	���
@��@��A㙚C�)���
@L(�@�\)BQ�CxR                                    BxnM̌  T          A
ff����@h��@33Ax��C�\����@>{@G
=A���C�                                    BxnM�2  �          AG���
@33=�?L��C$�{��
@��?.{@�C%u�                                    BxnM��  �          A���ff@	���}p�����C%���ff@z��G��8��C$�=                                    BxnM�~  �          A  �@�\���333C$:��@(Q쿇����HC"�                                    BxnN$  "          A�
��{@�
=@���A�C�f��{@��@�33B�C�\                                    BxnN�  
�          A
ff��@�p��0����\)C
=��@�ff>��@L(�C�H                                    BxnN$p  �          A���ff@b�\>W
=?�ffC����ff@X��?���@��
C�                                    BxnN3  �          A�����@�@}p�A�{C�R����@tz�@�G�B �
C��                                    BxnNA�  "          A\)��G�@�ff@dz�A���C�\��G�@�p�@�\)A�{CW
                                    BxnNPb  �          Az���p�@���@p��A�z�C�R��p�@��@���A��C��                                    BxnN_  T          Aff����@���@<��A�
=C������@��@���A�Q�CW
                                    BxnNm�  �          A(�����@�ff@)��A�  C�{����@z�H@mp�A��RCB�                                    BxnN|T  T          A�
���H@��\@�
Ak�C
� ���H@�33@j=qA�CL�                                    BxnN��  �          Ap��޸R@���?��RAK�
C(��޸R@�Q�@P  A�z�C��                                    BxnN��  �          A�����@��@qG�AĸRC����@z�H@��
B  C&f                                    BxnN�F  �          A��=q@�\)@z�AK�C���=q@fff@B�\A��RC#�                                    BxnN��  �          A�� z�@�?n{@��C(�� z�@���@
=qAP(�C�                                    BxnNŒ  "          A\)��@ƸR?�  A'33C!H��@�33@Mp�A�G�C
=                                    BxnN�8  �          A�\�\)@�z�@��RA��B�#��\)@�\)@�\)B.�HB��                                    BxnN��  T          A  �:=q@�G�@��A�=qB�33�:=q@��
@��HB0�HB��
                                    BxnN�  T          A���p�@�논��W
=B�q��p�@�\?�\)A=G�B�=q                                    BxnO *  "          AG���Q�A
=��Q��B�\)��Q�A\)?�(�A8��B�                                    BxnO�  �          A�
���A��?�G�@�B�{���A ��@C�
A�=qB�                                      BxnOv  T          A�R���A��?:�H@��B����A�@2�\A�z�B���                                    BxnO,  �          A���  A	p��u��
=B�B���  A=q?���A0(�B�\)                                    BxnO:�  �          A(���@�z��{�S�C�H��@�
=��
=�{C��                                    BxnOIh  "          A
=��=q@ᙚ�
=�\z�C����=q@���z��VffC��                                    BxnOX  
�          A  ��\)@�z����+�C:���\)@��
�u��p�CL�                                    BxnOf�  T          A�\��z�@�(��fff����CT{��z�@�z�?L��@�(�CG�                                    BxnOuZ  
�          AG���@�  ?�ff@�  B�Ǯ��@��@6ffA�Q�B�p�                                    BxnO�   �          A�����@�=q?��@��
B�����@�=q@=p�A�=qB�G�                                    BxnO��  "          A���=q@�33>�=q?У�B�=q��=q@�@
�HAS�B�W
                                    BxnO�L  �          A���>{@ƸR@�G�B G�B�.�>{@��@�(�BQ��B���                                    BxnO��  �          A���(�@�=q@�ffBCB���(�@J�H@�{Bs�RB�.                                    BxnO��  �          A��:�H@陚@���A�  Bߙ��:�H@�p�@���B0�B�                                     BxnO�>  
�          A
=�J=q@�Q�@�ffB�
B�Ǯ�J=q@���@�(�B@G�B�{                                    BxnO��  T          A33�J�H@��@�B<(�B�(��J�H@dz�@���Bj  C��                                    BxnO�  "          A
=�-p�@θR@��
B1�B�=�-p�@��HA�Bd=qB��                                    BxnO�0  	�          A�
�5@�@�=qB=qBߏ\�5@�G�@�z�BC�B�Q�                                    BxnP�  
�          A=q�a�@�{@�p�B=qB��H�a�@�33@�{BA�B���                                    BxnP|  
�          A�H�Z�H@��
@��A�Q�B�W
�Z�H@���@�Q�B,  B�\)                                    BxnP%"  P          A�
�AG�@�Q�@�ffAۅB��)�AG�@�Q�@��B!p�B�{                                    BxnP3�  �          A���hQ�@��
@�  B(�B�  �hQ�@�(�@ڏ\B633B���                                    BxnPBn  
�          A{����@�R@��A���B�Ǯ����@�z�@�p�B {B��{                                    BxnPQ  
�          A�R�aG�@��@�\)A�
=B�ff�aG�@���@�
=B
=B��                                    BxnP_�  �          A���2�\@��
@��\A뙚B�G��2�\@ȣ�@ə�B*33B�                                    BxnPn`  "          A���(�A�H@k�A�\)B���(�@���@�(�B�HBΊ=                                    BxnP}  "          A  �+�A(�@5A��B��R�+�@��@�z�A��
B���                                    BxnP��  T          A(��L��A�R@z�Ahz�B�.�L��AG�@�A��HB�=q                                    BxnP�R  T          A�H�\Az�@'
=A�Q�B�8R�\@�(�@��
A��Bʊ=                                    BxnP��  
�          A�H�33A
=q@C�
A�  Bγ3�33@�(�@��\B�HB�{                                    BxnP��  "          A
=�z�A	@>�RA��B�(��z�@��
@��B G�B��)                                    BxnP�D  �          A{�@  A�@*�HA�Q�B�  �@  @��@��A�z�B�G�                                    BxnP��  T          A���I��A  ?���AABܸR�I��@�Q�@~{Aʣ�B�8R                                    BxnP�  "          A�G
=A�?��A��B�B��G
=A�@_\)A��RB��                                    BxnP�6  "          A���N�RA
=?aG�@��\B��f�N�R@�
=@:�HA�=qB�#�                                    BxnQ �  "          A�\�dz�A{��{��B�k��dz�@�
=?�A,��B�L�                                    BxnQ�  �          A
=�uA	p����N�RB�33�uA\)?˅A�B��)                                    BxnQ(  S          A\)����@�{@�  A�G�B�#�����@���@�z�BC                                     BxnQ,�  �          A��e�@�ff@ƸRB�
B�=�e�@��R@�
=BJB���                                    BxnQ;t  �          A3��g
=@�(�@�p�Bz�B�33�g
=@�{A
=qBL�
B�=                                    BxnQJ  �          A3\)�S33A{@���BB����S33@ϮA(�BE�RB��f                                    BxnQX�  	�          A4�׿�\A�
@�G�A��B����\@���@�ffB3��B͞�                                    BxnQgf  
�          A1G����A
=@���A��B�Ǯ���@��\@��B.Bę�                                    BxnQv  "          A/���A=q@�  Bp�B���@�\A�
BC��B���                                    BxnQ��  T          A!��У�@�\)@��B1�B�LͿУ�@���AQ�Bh��B��H                                    BxnQ�X  h          A6=q��A	G�@�Q�B�B�����@���A
�\BL��B�ff                                    BxnQ��  	�          A��  @�  @�
=B%ffBה{�  @�{A   BZ�HB���                                    BxnQ��  �          A�\�\)@���@�\)B_
=B��)�\)@QG�A�B���B�z�                                    BxnQ�J  
�          A(�׿�Q�@�33A   BF=qBнq��Q�@�p�A{B|��B��
                                    BxnQ��  
�          A,z��Tz�@�
=A�B@
=B�\)�Tz�@���A�RBpz�B��3                                    BxnQܖ  T          A:=q���
@أ�A��B7(�B��R���
@���A33Bd=qC{                                    BxnQ�<  "          A:{��33@�z�A	B>�HC O\��33@tz�ABg�C                                    BxnQ��  �          A9G�����@��ABHz�B�aH����@_\)A z�Bq\)Cff                                    BxnR�  �          A8z���p�@��
A	�BC�RC+���p�@U�A33Bj{C��                                    BxnR.            A-���5�A�\@љ�B�RB�#��5�@���A�HBG\)B���                                    BxnR%�  |          A3\)��A�R@�Q�B�
BЮ��@��HAz�B@
=Bר�                                    BxnR4z  
�          A4z��
�HA��@��B�BϨ��
�H@�(�A
=qBHz�B���                                    BxnRC   T          A2ff�-p�A�@�  B�\B�.�-p�@߮A  B>�B�ff                                    BxnRQ�  
�          A3
=�\A�@��\A�\)B��Ϳ\Ap�@�=qB*\)B�G�                                    BxnR`l  T          A2�\��(�A�\@�=qAͮB��H��(�Aff@�(�B��B��)                                    BxnRo  
x          A Q�k�A�R?�(�A ��B��f�k�A
=@�=qA���B�Ǯ                                    BxnR}�  �          A$Q쿇�A33@p�AG33B�����Ap�@��
A�(�B�=q                                    BxnR�^  �          A,��>B�\A'�@!�AY�B��>B�\AQ�@��A��
B��q                                    BxnR�  �          A/
=�A�\@�z�A��Bͳ3�@�
=@��B3��B�B�                                    BxnR��  �          A/���=qA�@��HA��
B����=q@��@�Q�B-ffBΞ�                                    BxnR�P  |          A*=q���
A�
@��\B��Bģ׿��
@�Q�@�=qB>�RBȮ                                    BxnR��  �          A3\)���RA�@��A�Q�B�녿��R@�\)A   B6p�B�=q                                    BxnR՜  T          A4z�E�A{@��A�\)B��E�A�
@�B*\)B��R                                    BxnR�B  �          A-��#�
A=q@�p�A�z�B���#�
A��@�B&(�B�#�                                    BxnR��  �          A,��<#�
A�@�A��
B��)<#�
A ��@�p�B&�B���                                    BxnS�  �          A4(���33A�@��RA�ffB�\��33A@�G�B(33B��)                                    BxnS4  �          A7\)>W
=A"ff@��A�
=B���>W
=A�
@�{B(��B�.                                    BxnS�  �          A9��?
=qA
=@��A�B���?
=qA{A�B4�B�p�                                    BxnS-�  �          A9�<#�
A\)@�ffB	33B���<#�
@���A	�BCffB�Ǯ                                    BxnS<&  �          A8(��#�
A�R@׮B  B�{�#�
@�{AQ�BGffBܔ{                                    BxnSJ�  �          A9��5�A33@�z�B\)B�(��5�@陚A  B=z�B�\)                                    BxnSYr  �          A9��N{A�@�BffB��
�N{@�RAQ�B<�B�{                                    BxnSh  �          A8���`  A�@��A��B�  �`  @�z�@��HB+��B�B�                                    BxnSv�  �          A8���o\)AG�@�=qA��
B��o\)@�Q�A�RB4{B�=                                    BxnS�d  �          A7�
�N{A��@�p�A�z�B�.�N{@�  AG�B2�HB�p�                                    BxnS�
  �          A8(��tz�A{@�(�A��B�G��tz�@�
=@�\B%p�B��                                    BxnS��  �          A6�\��(�A{@���A���B�Ǯ��(�@��@�p�BB���                                    BxnS�V  �          A6�H��33A�@�\)A�  B�k���33@�@�B'��B�                                    BxnS��  �          A6{���HA  @�(�A�z�B������H@��@�=qB��C��                                    BxnS΢  �          A7
=�ۅ@��
@�  A�z�C��ۅ@�
=@�33B��C��                                    BxnS�H  |          A5p���p�A�R@���AӮB��f��p�@ۅ@��B�C�\                                    BxnS��  �          A6�H���\A{@�z�A�\)B�\)���\@�\)@��HB {C��                                    BxnS��  �          A7
=���
A�@�\)Aٙ�B�
=���
@У�@��
B
=C0�                                    BxnT	:  �          A7
=���\A=q@�G�A�B�(����\@θR@�B"  C�                                    BxnT�  �          A7
=��
=A\)@�G�A�p�B�aH��
=@θR@�{B){CE                                    BxnT&�  �          A5�����A�@���A�\B�������@�\)@�{B*=qCQ�                                    BxnT5,  �          A4����
=A��@�=qA��
B�Ǯ��
=@љ�@�  B-(�B�k�                                    BxnTC�  �          A2�\��A�R@��
A�{B����@�ff@�\B+z�B���                                    BxnTRx  �          A3����RA
{@��A�ffB�R���R@��@�\B*��B��3                                    BxnTa  �          A3�
��  A{@���A��HB����  @�@�Q�B'�B���                                    BxnTo�  �          A2{���@�{@e�A�z�C�q���@�Q�@�A�Q�C	�q                                    BxnT~j  �          A1�ff@�  @*=qA]p�C5��ff@��H@���A�Q�C.                                    BxnT�  �          A0  @�A��@�z�B�HB�#�@�@��
A  BS
=B�\)                                    BxnT��  �          A-�@�A (�@׮B�
B�.@�@�Q�A	�BT�\B�aH                                    BxnT�\  �          A*{��{A	p�@��
B�
B��{��{@�
=A{BI�B��
                                    BxnT�  �          A"ff�QG�A�\@�z�A�{B߮�QG�@�  @��HB ��B�3                                    BxnTǨ  T          A{��z�@���@�RAh��C ��z�@�z�@��HAЏ\C�
                                    BxnT�N  T          A=q��33@�{@?\)A��HCT{��33@�p�@�  A�C��                                    BxnT��  �          A&ff���A��@�ffB��B������@�G�@�33BE��B݊=                                    BxnT�  �          A(  �޸R@��@�Q�B(ffB���޸R@�\)A�Bbp�B�B�                                    BxnU@  �          A((���(�@�G�@ٙ�B!�
B�녿�(�@�Q�A	p�B\�B�L�                                    BxnU�  �          A$z��mp�@�Q�@��Bp�B��mp�@��@�(�B7�B���                                    BxnU�  �          A'����@ȣ׽L�;�z�C�3���@�=q?�=qA�Cٚ                                    BxnU.2  �          A'���R@������"�RC�{��R@ƸR�u���C�
                                    BxnU<�  �          A-�33@�Q�.{�c�
C��33@�33?��@�\C��                                    BxnUK~  �          A*{��@�\)�aG����HCk���@��H?��
@���C\                                    BxnUZ$  �          A"�\�
ff@�\)���
��Q�C���
ff@��?���@�C��                                    BxnUh�  �          A ����@�<�>��C�=��@�  ?��@���Cp�                                    BxnUwp  �          A$���33@��ÿ��J�HC���33@�  ?E�@�G�C�=                                    BxnU�  �          A&ff���@�(������陚C����@�=q���.{C�\                                    BxnU��  �          A%G��33@��
��Q�� ��C�H�33@��H�u���C�                                     BxnU�b  �          A+
=��R@hQ��(���eG�Ch���R@�ff�˅�	G�C��                                    BxnU�  �          A ����?�\��G����RC(�{��@1��\(�����C"�{                                    BxnU��  �          A"=q��@N�R�\)�Lz�C ޸��@mp������p�CL�                                    BxnU�T  �          A�R���?�(���(��<��C)�����@������
ffC&��                                    BxnU��  �          A�R���?��?\)��C+&f���@
=q� ���p  C&�                                    BxnU�  �          A�?�  �Vff��p�C*c��@��5���\)C%@                                     BxnU�F  �          A=q�	��?��S�
��z�C*�f�	��@\)�4z�����C%aH                                    BxnV	�  T          A���\@�ff?�@�33C�3��\@�ff@ ��A���Ch�                                    BxnV�  �          A
=�|��@�@�B{B���|��@�\)@�z�B6G�B��{                                    BxnV'8  �          A�\�g
=@ڏ\@�=qB{B�Ǯ�g
=@�33@�BC�
B���                                    BxnV5�  �          A�����H@�G�@���A�33B����H@��@�G�B,�
B�#�                                    BxnVD�  �          A���xQ�@�\)@��Bz�B����xQ�@�(�@��B;��C xR                                    BxnVS*  �          A��%@��
@�=qB%�
B�.�%@���@��B]B�{                                    BxnVa�  �          AG��=q@��@�z�B+�B�Ǯ�=q@�G�@�=qBcB�                                    BxnVpv  �          A���`��@�=q@���B�\B�  �`��@��@��BI�
B�z�                                    BxnV  �          A33�p��@˅@�ffB�RB���p��@�p�@޸RBE�C �{                                    BxnV��  �          A����
@�ff@�(�BffB�.���
@���@�p�B>��Cٚ                                    BxnV�h  �          Az���
=@Ǯ@��
B{B����
=@��@�33B?=qC�q                                    BxnV�  �          A����H@��H@�G�B �B����H@��@�(�B1p�C��                                    BxnV��  �          A�H��ff@�{@�RA_
=C����ff@�(�@k�A��C�3                                    BxnV�Z  �          A����@�  @u�A��RC Ǯ��@�
=@�Q�BG�C�                                    BxnV�   �          A{���
@�z�@���A���B��
���
@�G�@�Q�B��C
                                    BxnV�  �          A(����
@��@��AΏ\B������
@�33@���B��C�                                    BxnV�L  �          Aff���@�\@!�Au��B�Ǯ���@��
@�A�ffCT{                                    BxnW�  �          A����\@�z�(��s�
C5���\@��H?�G�@�G�CxR                                    BxnW�  �          A���G�@\>�=q?�Q�C0���G�@���?�A>=qC�H                                    BxnW >  �          A���33@�p�?��Az�C
��33@�=q@>{A���C��                                    BxnW.�  �          A33��
=@��?\(�@�G�CQ���
=@��
@ffAh��C�{                                    BxnW=�  �          A(���@���?��H@���C����@�\)@+�A��
C�
                                    BxnWL0  �          A�R���H@���?E�@�C�q���H@�33@  A]�C�                                    BxnWZ�  �          A���  @��\?���@��HCz���  @�=q@"�\Ayp�C!H                                    BxnWi|  �          Az���ff@�33?\(�@�{CO\��ff@�p�@�RAX��C�\                                    BxnWx"  �          A�R���@�(�@!�At  C�\���@QG�@c�
A�(�CxR                                    BxnW��  �          A����@������� ��C� ��@�=q?u@���C5�                                    BxnW�n  
�          A��@j=q>�  ?��RC����@\��?�  @�\)C�q                                    BxnW�  T          A����\@U��+���Q�C� ��\@XQ�>�\)?�C0�                                    BxnW��  �          A����
@Dz῁G����HC!&f��
@N�R���B�\C =q                                    BxnW�`  �          A��z�@N�R��(��>ffC���z�@i����  ��Q�Cn                                    BxnW�  �          AG���H@(��
=�e�C&:���H@0�׿�z�� ��C"ٚ                                    BxnWެ  �          A(���@ff�2�\����C&�H��@2�\�ff�L��C"s3                                    BxnW�R  �          A��
{@
=q�E����HC%�R�
{@;��
=�hQ�C!B�                                    BxnW��  �          A�
��H��z���  ��G�C?(���H��{��G����C6T{                                    BxnX
�  �          AQ���>�(����H��C1  ��?�Q���G��ڸRC(@                                     BxnXD  �          Az���ff?���{����C%�\��ff@I���~{���HC�                                    BxnX'�  �          AG���Q�?˅��{��RC'����Q�@>�R��G���ffC��                                    BxnX6�  �          A�R��Q�?�Q�������C$����Q�@Z�H��G�����C                                    BxnXE6  �          A��׮?����H��C$��׮@^�R�����HC�3                                    BxnXS�  �          A�\����=��
��Q��Ez�C3@ ����?�(��θR�:33C!33                                    BxnXb�  
�          A
=��(��z������Z��C:ff��(�?�  �����T��C#��                                    BxnXq(  �          A(���{�����=q�[�C?�\��{?����=q�[�HC()                                    BxnX�  �          A����H��G��陚�M
=C8B����H?����(��F\)C$E                                    BxnX�t  �          A$����@�G����8��CW
��@��
���R��p�C�=                                    BxnX�  �          A'33���@��&ff�c�
C����@��H?��\@�  Cz�                                    BxnX��  �          A,Q�����@陚���
�ڏ\C�\����@�?Y��@���C��                                    BxnX�f  �          A,Q��޸RA
=�Ǯ�CY��޸R@�
=?���A'\)C#�                                    BxnX�  T          A,(�����A�?
=@HQ�C�H����@��
@8Q�Ax  C�3                                    BxnXײ  T          A+\)�ƸRA ��@*�HAk�B�\)�ƸR@�ff@�33A�(�C��                                    BxnX�X  �          A,���.�R@�@�\)B-  Bݮ�.�R@�=qA�BiQ�B�#�                                    BxnX��  �          A*�R�+�@�p�A��Br�HB��
�+�@�\A&=qB�{BԸR                                    BxnY�  �          A1����Q�@��ABi33Bˮ��Q�@/\)A)p�B��B��
                                    BxnYJ  T          A2�H�8Q�@�Q�@�RB(
=BܸR�8Q�@�G�A�Be  B��                                    BxnY �            A5��{A{@�G�B!Q�B�{��{@�A=qBb  B��H                                    BxnY/�  �          A4�Ϳ�@�33A�BMffB�Q��@���A&�RB���B���                                    BxnY><  h          A4(��(��A	�@�z�B��B�\�(��@ʏ\A��BP��B�=q                                    BxnYL�  h          A0Q��0��Az�@w�A���BӔ{�0��A�
@���B��B��                                    BxnY[�  �          A1��E�@�p�A��B;��B���E�@���A(�Bv\)B�ff                                    BxnYj.  �          A4  �4z�@�=qA�Bu\)B�aH�4z�?\A+�B��RC�)                                    BxnYx�  |          A9p��w�A�H@�  B�HB��
�w�@�\)A	�B?ffB�Ǯ                                    BxnY�z  �          A7��a�A\)@��HB�B�#��a�@�p�Ap�BIffB���                                    BxnY�   �          A7��eA\)@�  B
p�B�q�e@�{A(�BG��B�8R                                    BxnY��  �          A7
=�hQ�A�@�p�B�
B�.�hQ�@�
=A
=BF
=B                                    BxnY�l  �          A2ff�G�A��@���A�G�B�#��G�@��@�B%��B��H                                    BxnY�  �          A5G��8��A��@�33B��B�k��8��@�33A\)BZ��B�\                                    BxnYи  ,          A8(��P  AQ�@�z�B��Bܮ�P  @�A�RBLQ�B鞸                                    BxnY�^  
�          A:�\�G
=A�R@ÅA���B؊=�G
=@�RA	B>z�B��                                    BxnY�  �          A9���@  AG�@�(�A�
=B���@  @��A ��B/\)B���                                    BxnY��  �          A:�\�-p�A�@�G�A�G�BҨ��-p�@���A33B3=qB�G�                                    BxnZP  �          A:�\�8Q�A�
@�\)A�p�B���8Q�@�A��B6��B�p�                                    BxnZ�  �          A;
=�%A�@�p�A��B����%@���Az�B;�HB��f                                    BxnZ(�  �          A;�
�p�AG�@�Bz�BΞ��p�@�AffBN  B�Q�                                    BxnZ7B  �          A:�\�.{A
{@�=qB�HB�
=�.{@�=qA��B\��B�\)                                    BxnZE�  �          A:{�Y��A�@�
=B�B���Y��@�z�AG�B^  B�=q                                    BxnZT�  �          A8���8��A33@��B33B��8��@�z�A�
B]p�B�B�                                    BxnZc4  �          A:{�<��A�
@�B=qB�W
�<��@��A��B]p�B��                                    BxnZq�  �          A7
=�-p�@�ffA�B=�RB�33�-p�@��
A"=qB|�B�{                                    BxnZ��  �          A9���0��@�{A{BGB�u��0��@}p�A(��B���B��)                                    BxnZ�&  T          A8����(�A
=@�ffB  B�#���(�@�G�A�HBJ��B��R                                    BxnZ��  T          A8(����A(�@�{A�33B�G����@���@�\)B/�
CL�                                    BxnZ�r  �          A9G���
=A��@���A�z�B�L���
=@أ�@���B)�B�=q                                    BxnZ�  �          A:ff����A(�@���A�p�B�B�����@�\)@�G�B  C ��                                    BxnZɾ  �          A:�H���
A�\@�G�Ạ�B� ���
@�
=@�  BC �H                                    BxnZ�d  �          A<  �ƸRAG�@�A�\)B�33�ƸR@�  @�(�B33C�\                                    BxnZ�
  |          A:�H���HA�H@z=qA��B��q���H@�\@�BQ�C޸                                    BxnZ��  �          A9�����HA�\@��\A�B�����H@��@�(�B  B�\                                    Bxn[V  T          A<(�����A�\@�z�A��B�G�����@�{AffB6�B��q                                    Bxn[�  �          A<����=qA�@�33A�G�B�\��=q@�A	�B<
=B���                                    Bxn[!�  �          A>{�@��A%�@��HAʣ�BԀ �@��Az�A   B)  B��                                    Bxn[0H  �          A;��;�A'\)@���A�=qB�L��;�A	�@�z�B=qBُ\                                    Bxn[>�  �          A>ff�XQ�A%�@�A�  B���XQ�A�H@�(�B (�Bߨ�                                    Bxn[M�  T          A>=q�%A.ff@~�RA�ffBή�%A�\@�(�BffBӅ                                    Bxn[\:  �          A<����Q�A (�@���A�C ��Q�@��R@�33B%=qCff                                    Bxn[j�  �          A>�R�
�H@��H@�p�A��C8R�
�H@`  @�33B��C�                                    Bxn[y�  �          A>�\�(�@��@�Q�A�(�C�q�(�@8Q�@��B��C!��                                    Bxn[�,  �          A?�
�	G�@�{@��
B G�C�H�	G�@Mp�@��B �RC�                                    Bxn[��  �          A?���@��@��\A��C�
��@`  @�  B	p�C=q                                    Bxn[�x  �          A>�R�@��@���A���C)�@QG�@�z�BG�C                                     Bxn[�  
�          A@  �ff@���@�G�A�G�C���ff@8��@�Q�Bp�C#T{                                    Bxn[��  �          A@���{@��@���A�Q�C�q�{@7
=@�  A�(�C#ٚ                                    Bxn[�j  |          A=p��˅A
�R@���A��B���˅@�  @�B�HCJ=                                    Bxn[�  �          A=p���=qA\)@��A�33B�\��=q@�G�@陚B(�C �{                                    Bxn[�  �          A<Q�����AQ�@���A�  B랸����A Q�@��
B33B�p�                                    Bxn[�\  T          A;�
���A"�R@l��A�G�B�3���A  @�ffB�B�                                    Bxn\  �          A=�����HA'�@[�A���B�\���HA@љ�B=qB�8R                                    Bxn\�  �          A=���=qA+
=@C33An�RBݮ��=qA
=@�Q�A�33B�Ǯ                                    Bxn\)N  �          AAp���A2�R@\)A�{Bˏ\��A��@��B�RB���                                    Bxn\7�  �          AA���(�A8z�@8Q�A]p�B�k���(�A z�@�(�A���B�=q                                    Bxn\F�  �          AC\)�
=qA=��?�=qA
=qBȮ�
=qA*�\@�\)A�  B��                                    Bxn\U@  �          AD���<��A;�
?��AffB�=q�<��A(z�@���A�{B�W
                                    Bxn\c�  �          AA�	��A.ff@�33A���B�\)�	��A��A ��B&Bυ                                    Bxn\r�  �          AC
=��  A4��@�A�33B�#׿�  A=q@��B  B�.                                    Bxn\�2  �          AAp����
A5�@~{A�p�B�W
���
A(�@�z�B�B�\)                                    Bxn\��  �          A?��{A*{@�  A��
B˔{�{A�A{B+p�B�W
                                    Bxn\�~  �          AAG��A&�H@��RA�Bʣ��A ��A  B9��B�\                                    Bxn\�$  T          A>�\��RA�@���A�p�B�����R@�Q�A�\BA��Bؔ{                                    Bxn\��  T          A>�R�8Q�A33@ə�A�33B��8Q�@�p�AG�BFp�B߸R                                    Bxn\�p  �          A?�
�{Aff@��A�
=B�Q��{@�33A�\BH(�Bծ                                    Bxn\�  �          A@Q��S33A�@�z�AǮBؙ��S33@��@��HB*\)B�                                     Bxn\�  �          AC\)�
ffA	�>�Q�?ٙ�C{�
ffA ��@G�Am�C	�                                    Bxn\�b  |          AD  ��RAG�@��A5G�B����RA��@��
A�ffC�                                    Bxn]  �          A=p���z�A�R@
=A$  B�ff��z�A�@���A�p�C��                                    Bxn]�  �          A<���׮A�?���A=qB��
�׮A�@�G�A�Q�C ��                                    Bxn]"T  �          A<Q��߮A
=@
�HA*ffB�z��߮@��@�G�A�G�C.                                    Bxn]0�  �          A<(���=qAz�@A7�
B�����=qA (�@�\)AӮCff                                    Bxn]?�  �          A>�H��{A#�@.{AS�B�
=��{Az�@��A�=qB�                                    Bxn]NF  �          A>�H���
A((�@�A��B������
AQ�@��A֏\B��)                                    Bxn]\�  �          A<(���Q�A+�@ffA$��B�W
��Q�A33@�  A߅B�\                                    Bxn]k�  �          A=���A)��@�A5B�33���A  @�A�33B�G�                                    Bxn]z8  �          A@(����A!��@3�
AX��B������A	�@�
=A�
=B���                                    Bxn]��  �          A@�����A%�@7�A\Q�B�
=���A��@�33A�B��                                    Bxn]��  �          A@z���\)A#
=@0  AS�
B����\)A�@��RA��B���                                    Bxn]�*  �          A@Q�����A'�@p�A+33B�p�����Aff@���A�B��H                                    Bxn]��  �          A<����Q�A�\@(��AO�
B�W
��Q�A (�@��A�{C)                                    Bxn]�v  T          A=G����RA
=@
=A8Q�B��H���RA	p�@���A޸RB�u�                                    Bxn]�  �          A>{��A��@%�AI�B�u���A{@�{A�G�B�Ǯ                                    Bxn]��  �          A=�����A�@�\A�HB�\���A�@�  A�=qB��\                                    Bxn]�h  �          A<�����A!�?�Q�A�B�q���A�@�
=A�33B��
                                    Bxn]�  �          A;�
��z�A ��?�ff@�(�B���z�Az�@��\A�(�B�33                                    Bxn^�  �          A:�\���
A Q�?G�@u�B�q���
A�H@��\A�ffB�.                                    Bxn^Z  �          A:�H��p�A?J=q@w�B����p�Az�@���A�ffB��q                                    Bxn^*   @          A6�H��  Aff���
�У�B�3��  A33@<(�Ao�B�33                                    Bxn^8�  �          A8����RAz�>�  ?�G�C ����RA�R@Q�A�C��                                    Bxn^GL  �          A9���33A�?�@"�\CG���33Az�@`��A���C��                                    Bxn^U�  �          A<(���{Ap�=�G�?��CO\��{AQ�@K�Az�HC&f                                    Bxn^d�  �          A=����p�A�=�\)>��
C �\��p�A
�R@L(�Ay��C�
                                    Bxn^s>  |          AA�����A�\>��R?�p�B�8R���A�
@dz�A���C!H                                    Bxn^��  h          A=����\A
�R@��A��B�{���\@�AffB@Q�B�                                    Bxn^��  �          A<���j�HA�@�B\)B��H�j�H@�  A
=B^��B�k�                                    Bxn^�0  �          A<(���z�A��@�\)A�33B��f��z�@���@�33B(�B�G�                                    Bxn^��  �          A<z�����A33@�=qA�  B�\)����@��R@�
=Bz�B�                                      Bxn^�|  �          A;\)��{A\)@�Q�A�B�#���{@���@��
B"p�B�z�                                    Bxn^�"  �          A<  ��=qA�R@��RA�p�B�{��=q@�  @��B ��C ��                                    Bxn^��  �          A<(����A�H@p  A��B�aH���@��@�33B�B�p�                                    Bxn^�n  �          A;�����AQ�@��HA��B�k�����@�Q�@�ffB��B�Ǯ                                    Bxn^�  @          A:=q����A�\@^�RA���B��q����@���@�p�B�C.                                    Bxn_�  |          A<Q���G�Az�@+�AS�C�3��G�@�=q@���A��C�f                                    Bxn_`  �          A;��ƸRA\)@��A���B��\�ƸR@���@�{B p�C�3                                    Bxn_#  �          A<z���
@�Q�?�R@AG�C���
@ڏ\@K�A{�C                                    Bxn_1�  �          A:�R�(�@��
�O\)��  C���(�@�{?��AG�Cff                                    Bxn_@R  �          A:�R�	G�@�p����ÿ�\)C	G��	G�@�G�@(�AA�C
�                                    Bxn_N�  �          A;��	G�@�����ÿ�\)C	
=�	G�@�33@{AB�RC
p�                                    Bxn_]�  �          A9����@�\)�}p���(�C�3��@�33?�
=Ap�C.                                    Bxn_lD  �          A:=q�z�@�=q�(��?\)C�R�z�@�=q?��RAp�C�                                    Bxn_z�  �          A:�R��@�=q�#�
�W
=C����@�33@'
=AN�HCT{                                    Bxn_��  �          A:ff�ff@����
��p�C���ff@ᙚ@#�
AK33C�H                                    Bxn_�6  �          A9��ff@��ÿ���0  C��ff@\?�Q�A�HC�\                                    Bxn_��  �          A9���(��@�\)�!G��G
=C�
�(��@���?�=q@��HC5�                                    Bxn_��  �          A8���(��@��\���H����C�=�(��@��R>�@Q�C�                                    Bxn_�(  �          A7�
�"�H@�Q�\(���G�C���"�H@�
=?�ff@�Q�C�{                                    Bxn_��  �          A8  ���@�Q쿪=q��(�C�����@�33?L��@~�RCJ=                                    Bxn_�t  �          A7�
�33@��ÿ�G�����C#��33@\?u@���C��                                    Bxn_�  �          A6�R���@�\)���R��ffC�{���@�{?�
=A�C�                                    Bxn_��  �          A7\)�z�@�z�˅��C@ �z�@�=q?(�@FffCz�                                    Bxn`f  
�          A8Q����@�\)��(��	CJ=���@�p�?333@^�RC�                                    Bxn`  T          A8�����@�  @'
=A\(�CY����@��@�ffA�\C=q                                    Bxn`*�  |          A9��u�A��@��A�p�B��)�u�@�{Az�BB�
B�\                                    Bxn`9X  �          A6ff��{Aff@�z�A��B�#���{@�
=@��
B!G�B�\)                                    Bxn`G�  �          A6�H��  AG�@�Q�A��HB��
��  @���@��
B&��C)                                    Bxn`V�  �          A8z���\)A\)@��A�(�B� ��\)@��A�
B=  Cc�                                    Bxn`eJ  h          A9p���G�A��@�(�B �B�.��G�@��HA�BD��C�                                    Bxn`s�  �          A9���Q�A=q@���A��B�#���Q�@�p�A��B?��Cs3                                    Bxn`��  |          A:�R���
@�
=@�Q�BC k����
@�A
{B>�C#�                                    Bxn`�<  h          A8����G�A(�@�p�A�ffB����G�@�=q@�G�B�Ch�                                    Bxn`��  T          A8  ���A z�@QG�A���B�z����A�\@�{B(�B���                                    Bxn`��  �          A5G����A33?�Ap�B��f���A	��@��\A���B��                                    Bxn`�.  �          A6{����Ap�@)��AX  B�G�����@�  @���A���C�                                    Bxn`��            A8z��љ�A��@,(�AX  B�� �љ�@�@�=qA���C+�                                    Bxn`�z  T          A6�\��(�A�
@\)AJ�RB�{��(�@�{@�ffA�RC �=                                    Bxn`�   �          A7�
��ffAG�@��AF=qB�z���ffA Q�@�G�A�B�Ǯ                                    Bxn`��  T          A8z���G�A�@A<(�B�����G�@��R@���A�  B�W
                                    Bxnal  �          A:�\��=qA��@k�A�  B�����=q@��H@ۅB�
C k�                                    Bxna  �          A;���33A��@��HA�Q�B�\��33@ᙚ@�RB�\C n                                    Bxna#�  T          A;�
����A�@�(�AŅB��=����@��@�  B&=qC�{                                    Bxna2^  �          A<z���A�@��A�(�B�����@��A33B?
=C�R                                    BxnaA  �          A;����\A�@��RA�z�B��)���\@\A	�B=
=C�)                                    BxnaO�  h          A?
=����@�G�@߮B33B������@�p�A��BOz�C��                                    Bxna^P  �          A?\)��\)@�=q@�=qB\)CaH��\)@vffA�RBP��CJ=                                    Bxnal�  �          A@����z�@��@�\)BC�\��z�@S�
AffBU33C�H                                    Bxna{�  �          A@���޸R@�{@�=qA�ffC
�޸R@���A
=qB8Q�C��                                    Bxna�B  �          A?�
�׮@�=q@�\)A��\C���׮@�ffA	B9G�C)                                    Bxna��  �          A?33���R@���@��B��C #����R@z�HA�BZ(�C�                                    Bxna��  h          A?����
@�{@�33B�
C�����
@��\A{BGffC��                                    Bxna�4  |          A@  ���
@�Q�@�{B�CL����
@��RAQ�BD�C
=                                    Bxna��            A@  ���A�\@�G�A���B�ff���@�G�A
�RB9�
C                                      BxnaӀ  @          A@z����@�33@��
B�HB��f���@�Q�A
=BW�HC�                                    Bxna�&  |          A@z�����@�
=@��HB�Cz�����@�  A=qBM�
Cff                                    Bxna��  �          A@z���z�@�=q@أ�B�C &f��z�@�{A�BH(�C��                                    Bxna�r  �          A<����{A
=@�p�A�G�B��f��{@ȣ�@�B#Q�C��                                    Bxnb  �          A:�R�ə�AQ�@P  A�\)B�u��ə�@��H@У�B�HC�f                                    Bxnb�  �          A;�
�ȣ�A�
@c�
A�G�B�\)�ȣ�@�ff@ٙ�B��C{                                    Bxnb+d  �          A;\)��ffA�H@a�A�=qB�����ff@�p�@��B
��C)                                    Bxnb:
  �          A:�H��p�Aff@>�RAmG�B�����p�@��H@���A��\CT{                                    BxnbH�  �          A<  ��A\)@G�A333C���@�{@�A��HC0�                                    BxnbWV  |          A;�����A��@J�HA|z�B�
=����@�z�@ϮB(�CaH                                    Bxnbe�  �          A<(����RA\)@�\)A��B�����R@�33@��
B"�RB�
=                                    Bxnbt�  T          A;����HAff@5A`��B����H@��H@�\)B ��C�H                                    Bxnb�H  �          A;��ۅA��?���Ap�B��q�ۅ@�z�@�G�A֣�C�                                    Bxnb��  �          AB{�  A33��G���C}q�  A\)?�(�A�\Cu�                                    Bxnb��  |          A>�R���
A�@���A��B��q���
@�Q�@�B�\C.                                    Bxnb�:  �          A>{���A�@(�A+\)B�\���@�  @�(�A�C�3                                    Bxnb��  �          A=���
A  ?#�
@ECG����
A ��@�=qA��Cs3                                    Bxnb̆  �          A@����
A�
�8Q��[�CL���
Ap�@.{AQG�C��                                    Bxnb�,  |          AA����A�
?���@�  CE��@��@�
=Ař�C�{                                    Bxnb��  h          AB=q�l��@��HA{B0�B�\�l��@}p�A,Q�B}{C�                                    Bxnb�x  �          AAp��|(�A�
@�  Bp�B�{�|(�@�
=A!G�Bd{B��                                    Bxnc  �          AA������@���A(�B,ffB�
=����@|��A*ffBv�
C^�                                    Bxnc�  �          AA��A z�A ��B&�B�  ��@�  A(��Br�C��                                    Bxnc$j  �          AB�\���A@���BffB�B����@��A#33Bd�RC�f                                    Bxnc3  �          AA�����A�@��B��B�=����@��
A
=BT�C^�                                    BxncA�  @          A>=q��{A  ?L��@��HC8R��{@�  @�=qA��C�                                    BxncP\  �          A=���
@��R��p��\)C�{��
A   ?��@���C��                                    Bxnc_  �          A=��ffA  �E��p��C�=�ff@�(�@"�\AF�HC�{                                    Bxncm�  �          A<���(�@�
=���� (�C���(�@�
=?��@��C��                                    Bxnc|N  �          A;�
��{A{�z�H��=qC����{A��@'
=AO�
C\                                    Bxnc��  |          A9G����\A  @\��A�
=B������\@�@�B�\B�G�                                    Bxnc��  �          A9��  A\)@(Q�AR=qB뙚��  A��@˅B�\B��f                                    Bxnc�@  �          A9����A�
@<(�Ak�B������A   @�p�BB�\                                    Bxnc��  �          A9�����\Az�@`��A���B�k����\@�33@��B��B��                                    BxncŌ  �          A<Q��`��A+�@R�\A��
B�B��`��Az�@�G�BG�B�                                    Bxnc�2  �          A<������A\)@�p�AڸRB�33����@�Q�A  B?=qB��R                                    Bxnc��  �          A=G��XQ�A��@��HB	�B��f�XQ�@��A�\B^�RB�                                    Bxnc�~  �          A;\)�c�
@�G�A�\B8{B�
=�c�
@S�
A)�B���C	\                                    Bxnd $            A<Q��dz�A��@�z�B��B�B��dz�@�A!�BoG�B���                                    Bxnd�  �          A<���r�\A��@�ffB�B��H�r�\@�p�A Q�Bh�
B�.                                    Bxndp  �          A=��+�A
�\@�33B�HB�k��+�@��
A'33Bw�B��3                                    Bxnd,  �          A=p��'
=A	�@�(�B!{Bճ3�'
=@��A'33Byp�B��                                    Bxnd:�  �          A<������A�\@�B=qB�ff����@���A\)Bg(�C+�                                    BxndIb  �          A>=q�L(�A��@��B��B����L(�@�  A%�Bt\)B�Ǯ                                    BxndX  h          A=���AG�@љ�B	(�B�p���@��
A(�Bd�B�aH                                    Bxndf�  �          A>ff�xQ�A%p�@�\)A�33B��3�xQ�@��
A�HBRBÀ                                     BxnduT  �          A<z���@�33ABD��B�33��@G
=A1G�B�u�B��\                                    Bxnd��  h          A<Q�>k�@�{AQ�BU�B�\>k�@(�A/\)B���B�33                                    Bxnd��  �          A<z�?���@�{A#�Bv(�B�?���?L��A8��B��
B�                                    Bxnd�F  �          A;�=u@��A/\)B�Q�B�p�=u�E�A:�RB�p�C��                                    Bxnd��  T          A<z�>aG�@mp�A1�B�  B�33>aG���z�A:�RB�u�C�b�                                    Bxnd��  �          A<Q��
=@�G�A%��Bx�B� �
=>�Q�A8  B��fC*p�                                    Bxnd�8  �          A;
=���R@�=qA)�B�(�B�Q쾞�R<�A:�\B���C.{                                    Bxnd��  �          A:�R���R@�z�A��Be��B��Ϳ��R?���A733B�.C 8R                                    Bxnd�  �          A;33���@�=qA,��B�B�k���녿   A9�B���CK��                                    Bxnd�*  �          A;
=>���@w
=A0  B�=qB�.>������\A9�B��\C�"�                                    Bxne�  �          A:�H��@�p�A,z�B�#�B�Ǯ����(�A:=qB�G�CZ��                                    Bxnev  �          A<(���\@�ffA+�
B�� B�W
��\�#�
A;�B���CE��                                    Bxne%  |          A<����
@߮A�BLp�B؅��
@%�A333B���C�                                     Bxne3�  �          A>�R��p�A��A ��B*  B�#׿�p�@�{A-B��qB�#�                                    BxneBh  h          A=녿�Q�A�A�
B1  B�33��Q�@�{A/�B���BӸR                                    BxneQ  h          A=�����Az�@�
=B*ffB�ff���@�{A,��B���B�#�                                    Bxne_�  �          A<�ÿٙ�A
=@�ffBp�BɅ�ٙ�@�(�A$Q�By{Bخ                                    BxnenZ  |          A<Q��:�HA�R@�33B
=Bր �:�H@��A33B_(�B��                                    Bxne}   �          A=�h��A�H@�Q�A��B�B��h��@�(�Az�BN�B�p�                                    Bxne��  �          A=p����A#�
@���A�p�B������@��
@�{B)G�B�33                                    Bxne�L  �          A<������A!�@|��A�\)B�  ����@�G�@��\B'�B�{                                    Bxne��  �          A<������A\)@���A��
B�����@�33@��\B&�HB�W
                                    Bxne��  �          A:�\��\)A�R@�z�A�(�B�R��\)@��HA�RB2�B��                                    Bxne�>  �          A:{��{A'33?��@��B�ff��{A=q@��A���B��                                    Bxne��  �          A7����
A
=>8Q�?k�B�  ���
A�H@��A��B�aH                                    Bxne�  h          A9���33A
�R@�
=A�B�u���33@�{A��BJ(�C��                                    Bxne�0  �          A7���Q�A#
=@(�AE�B�R��Q�A33@���B�\B홚                                    Bxnf �  h          A6{���
A#33@��A0(�B�  ���
AG�@�Q�B�
B螸                                    Bxnf|  �          A7
=��  A��L�Ϳ�G�B�B���  A@���A�z�B�                                     Bxnf"  �          A8�����Az�#�
�N{B�  ���A{@dz�A���B��{                                    Bxnf,�  |          A6�H��G�A$(�>�(�@(�B�
=��G�A��@�Q�Aƣ�B�                                    Bxnf;n  T          A7
=����A$Q�?�  @�G�B�������A@�G�A܏\B��
                                    BxnfJ  �          A6�R��A%�����;��HB�\)��Aff@��\A���B�z�                                    BxnfX�  |          A6=q�^�RA+���G����
B����^�RA"{@j�HA�\)B���                                    Bxnfg`  �          A6�R��Q�A%G���  ��B�.��Q�A!G�@5Af�RB�33                                    Bxnfv  �          A6�R�~�RA(�ÿ�G���(�B�G��~�RA#
=@J=qA���Bި�                                    Bxnf��  �          A6�R���\A�\���
=B�{���\AQ�@!G�AL  B�=q                                    Bxnf�R  �          A6�R��(�A��u���33B����(�A  >���@ ��B�=q                                    Bxnf��  �          A6{���RA  �,(��\��B��)���RA\)?�\A��B��f                                    Bxnf��  |          A5���L��A-�����H�p�B��f�L��A   @��A��Bׅ                                    Bxnf�D  h          A5p��X��A*�\�s33���HB�Q��X��A Q�@p  A�  B�ff                                    Bxnf��  �          A5p����A0�Ϳ=p��o\)Bʏ\���A$��@��
A���B�33                                    Bxnfܐ  �          A4�����RA���(���a�B�����RA  ?�  A��B�33                                    Bxnf�6  T          A7\)��  A���Q�����C����  A��\)�4z�B�W
                                    Bxnf��  �          A8�����R@�p��U���RC\���RA>��
?�=qC�
                                    Bxng�  �          A8Q����HA Q��%�PQ�CY����HA��?�33@�  C33                                    Bxng(  �          A8Q���p�A33�33�8��Cu���p�Ap�?�(�A	G�C�                                    Bxng%�  �          A3�
�У�A���=p��v=qB���У�A�?��@�\)B���                                    Bxng4t  �          A/\)��ffA��@  �~{B�k���ffA��?�
=@��B�
=                                    BxngC  �          A/����A
ff�J�H��33B�=q���A�R?n{@�(�B�.                                    BxngQ�  �          A-G��ٙ�A  ��
=�ə�C���ٙ�@�@#33A[�C�)                                    Bxng`f  �          A.�H��33A�\�   �,Q�B����33A
=?��A"=qB��                                    Bxngo  �          A+
=���@�z������1\)C�H���A���p����B�                                    Bxng}�  �          A+
=��p�@�=q��p��!�C:���p�@�=q�xQ���Q�C @                                     Bxng�X  �          A,z���p�@������
�
�C)��p�@�  �X�����HC�f                                    Bxng��  �          A-��c�
A�H��\)� =qB��
�c�
A!p���
=���HB��f                                    Bxng��  T          A,  �'
=A=q�����p�B��
�'
=A&�\��G���B�#�                                    Bxng�J  �          A+33�\)Az��������B�B��\)A&=q>��H@'�B��                                    Bxng��  �          A+33�c33A��]p����B���c33A Q�?��@�33B�                                    BxngՖ  �          A+��vffA33�R�\����B�W
�vffA�H?�  @�p�B�ff                                    Bxng�<  �          A+���{AG��e���(�B�\��{A�?Q�@���B�                                    Bxng��  �          A)����HA(��J=q����B�\)���HA  ?��@�  B��                                    Bxnh�  �          A+
=����A���dz���(�B�(�����A��?�@1G�B��
                                    Bxnh.  �          A)��ǮA�
�p��A�B�B��ǮA?ٙ�A�
B�u�                                    Bxnh�  �          A(���ȣ�@�{�z=q��z�C��ȣ�A�H��z�У�B��                                    Bxnh-z  �          A)p���@���������C	����@��H�����Q�C}q                                    Bxnh<   �          A)���  @�33��Q���=qC�\��  A=q�����RC�f                                    BxnhJ�  �          A(���Ϯ@ᙚ������33C�)�ϮAz�E���C �                                    BxnhYl  T          A)��\)@�{��
=��RC)��\)A\)��Q����HC L�                                    Bxnhh  �          A*�H�ҏ\@׮��{�݅CT{�ҏ\A�
�����  C ��                                    Bxnhv�  �          A*=q�׮@�{��ff�ř�C33�׮A33�Y����33Cs3                                    Bxnh�^  �          A'���\@���������\Cٚ��\@�  �Mp���
=CQ�                                    Bxnh�  �          A(����{@�(���33� ffC���{@��p��W�
C��                                    Bxnh��  �          A'33����@�{���H��HC	�
����@�=q�7
=�
=C �                                     Bxnh�P  �          A*�H����@��H��ff�
=C� ����A��G
=���B�B�                                    Bxnh��  �          A*{�ə�@�33�ȣ��ffC	!H�ə�A ���<(���Q�C 
=                                    BxnhΜ  �          A)��θR@ȣ������\)C�
�θRA{�	���;\)C s3                                    Bxnh�B  �          A*{���@�  ��=q��RC�\���A�ÿ��H��z�B�L�                                    Bxnh��  �          A+�����@�33�P  ���CxR����@���>���?�
=B���                                    Bxnh��  ,          A8�����
A�@Q�AD��B������
@�(�@�=qB	�\Cz�                                    Bxni	4  
�          AC
=��\)AQ�?�=q@���B�k���\)@�\)@���AхC(�                                    Bxni�  ,          AHQ�����AQ�c�
��(�C5�����A{@c�
A�=qC0�                                    Bxni&�  �          AQp���Q�A��@���A��B�
=��Q�@�33A!�BHCǮ                                    Bxni5&  �          AP����A�R@�z�A�  B�q��@�(�Az�BC��C\                                    BxniC�  �          AU����HA�H@ٙ�A���B�G����H@��RA)��BX�RC�R                                    BxniRr  �          AQ�����A"ff@��A���B�u�����@ҏ\A�B6ffC0�                                    Bxnia  �          AP(���G�A��@��\A�
=C Q���G�@��
A33B"=qC�=                                    Bxnio�  �          AO���(�A  @�Q�A��
B�L���(�@�{AQ�B�RC	�\                                    Bxni~d  �          AO\)��
A�R@Tz�An{C&f��
@ڏ\@�G�B	
=CO\                                    Bxni�
  �          AO��G�A�R@mp�A���C�H�G�@���@�(�B\)C�=                                    Bxni��  �          AP����A�H@�p�A�(�B�u���@�A	�B%��C
�{                                    Bxni�V  �          AR{��(�A{@���A�G�B�����(�@�  A�\B*��C	�H                                    Bxni��  �          AR{���\A{@�G�A��\C!H���\@љ�A(�B�HC
                                    BxniǢ  
�          AR=q�G�A\)@���A���C)�G�@�p�AG�B�RCs3                                    Bxni�H  �          AQ����Aff@��A��C5����@\AB��C                                    Bxni��  �          AQ�� ��A{@��A�\)C��� ��@�G�A�B=qC�                                    Bxni�  �          AQp���z�A�\@���A�{C����z�@�
=A{BffC��                                    Bxnj:  �          AQp���p�A@�33A�z�C@ ��p�@�z�A�HB 33C5�                                    Bxnj�  �          AR=q�ᙚA��@�Q�Aƣ�B����ᙚ@�p�A��B5�C�R                                    Bxnj�  �          AR{���A(�@���A�(�B�.���@���A
=B7�C��                                    Bxnj.,  �          AS����A(�@�G�A�B����@�
=A (�BE�C��                                    Bxnj<�  �          AUG�����Aff@���BG�B������@�\)A2�\Bc��Cs3                                    BxnjKx  �          AT(����A=qAz�B�B�����@~�RA7�Bo�HC)                                    BxnjZ  �          AT(���  AQ�@�ffB(�B�ff��  @��\A2ffBe�C=q                                    Bxnjh�  �          AS
=����AG�@�(�B�\B�\����@�G�A5G�Bl��C	�)                                    Bxnjwj  �          AS���p�A(�A��B
=B���p�@���A;
=By{C��                                    Bxnj�  T          AUG��}p�A\)A33B��B�u��}p�@��
A>�\B~�\C0�                                    Bxnj��  �          AVff�\)A	p�A�B4  B��
�\)@G�AFffB��\C�q                                    Bxnj�\  �          AU����
A�AB7�B�����
@3�
AFffB�33C��                                    Bxnj�  �          AUp����H@��A�B>�HB�B����H@Q�AG�B�k�C�=                                    Bxnj��  �          AV=q�y��A(�A��B;��B晚�y��@)��AH��B��qC��                                    Bxnj�N  �          AV�H�q�A�RA  B:\)B�W
�q�@333AIG�B���Cp�                                    Bxnj��  �          AV�H�s33@��RA!BC��B�
=�s33@{AJ�HB��C��                                    Bxnj�  �          AV=q��  AA�B2
=B�\)��  @,��AAp�B���C�f                                    Bxnj�@  �          AV�H����@�(�A�B?B�  ����?�p�AF=qB�ǮCff                                    Bxnk	�  �          AV=q��z�@���A�B:�\B�W
��z�@{AD(�B���C�
                                    Bxnk�  �          AT�����A   AB'�B�z����@4z�A9��Br�
CL�                                    Bxnk'2  �          AP  ���A(�@��A���C�=���@��\A�RB=qC�H                                    Bxnk5�  �          AN�H���A�\)��RCff���A��@��HA�p�C�)                                    BxnkD~  @          AT����A�H��(��˅C���A(�@EAW33C	:�                                    BxnkS$  |          AQ����A ��?#�
@3�
C�=���AQ�@��A�(�C:�                                    Bxnka�  �          AR=q���A33>�Q�?�ffC�R���A��@�=qA��C�q                                    Bxnkpp  �          AR�H�=qA�H��\���C33�=qA�@L��Aa�CB�                                    Bxnk  �          AT  � ��A   �P���c�
C ��� ��A%p�?�
=A{B��                                    Bxnk��  �          AXQ��(�A33�$z��/�
C^��(�A\)@ ��A+�CQ�                                    Bxnk�b  �          AXQ����A�H�'
=�2ffC�����A\)@�RA(��Ch�                                    Bxnk�  T          AV�H��Az��ff���RC\��A�@@��AO�
C�                                    Bxnk��  �          AW\)��RAQ쿮{���C33��RAz�@QG�A`��C	�                                     Bxnk�T  �          AV�R��A���5�C\)C8R��A�@�AG�C�                                    Bxnk��  �          AU��
=A(��g��{33Ck��
=A!�?�@�=qC�                                    Bxnk�  �          AU��(�AG��j�H��  C�(�A�?�@���C�\                                    Bxnk�F  �          AUG��  A{�N�R�a��C޸�  A#\)?�Q�A�RC �                                    Bxnl�  
�          AUp����
A)G��Tz��f�RB�����
A-@{Ap�B�G�                                    Bxnl�  �          AU��p�A$���hQ��|z�B�L���p�A,z�?�@��B��H                                    Bxnl 8  �          AUG���A�
�w
=���HC(���A&=q?��@�G�C W
                                    Bxnl.�  �          AT�����A���|����(�C�����A$��?��H@��RC �H                                    Bxnl=�  �          ATQ��	p�A����  ��Q�C�q�	p�A!�?��\@���CxR                                    BxnlL*  �          ATQ����AG��vff��p�C)���A��?�ff@��C��                                    BxnlZ�  �          AW��$  @����p���(�C��$  A  ���� ��C��                                    Bxnliv  �          AW\)��RAG���
=��=qC���RA�8Q�J=qC�3                                    Bxnlx  �          AW
=�  @�������
C#��  A(�����   CxR                                    Bxnl��  �          AU��{@�����Q���{C��{A������C5�                                    Bxnl�h  �          ATz��33@��H������Q�CQ��33A�Ϳ\)��HCu�                                    Bxnl�  �          AR�H�G�Ap���(����C
O\�G�A�ÿ��G�C��                                    Bxnl��  �          AS
=�@�{��G���p�C���A�R�������C
=                                    Bxnl�Z  T          AR{��@�
=��
=�㙚Cc���A�ÿ�G���  CY�                                    Bxnl�   �          AP(��
{@�=q��33��p�C	�{�
{A�H��G���=qC��                                    Bxnlަ  �          AO
=�{@����G���
C��{A
=�^{�yp�CY�                                    Bxnl�L  �          ANff�G�@ʏ\��33�{C���G�A  �`���\)C+�                                    Bxnl��  �          AMG���R@�  �\)�&(�C�)��RAG�������z�C+�                                    Bxnm
�  �          AN�\��@�G���33��  C����A���  ��{C�)                                    Bxnm>  �          AP(��Q�@����ҏ\��
=C���Q�@�33�G��_\)C8R                                    Bxnm'�  �          AM��{@�p��Q��,Q�C)�{A=q��=q���C
                                    Bxnm6�  �          AMG��p�@S�
�{�6{CW
�p�@�z������(�C
s3                                    BxnmE0  �          APQ�����@�
=����7\)C}q����A  �����  C�H                                    BxnmS�  �          AO33��R@�{�\)�G�C
���RA{�tz���\)B��                                    Bxnmb|  �          AN{��z�@�����ffC(���z�A\)��
=���C�
                                    Bxnmq"  �          AO��
ff@��
��ff�p�C�)�
ffA�\�q���\)C(�                                    Bxnm�  T          AO33�
=@��
��33Cz��
=A���  ���HC��                                    Bxnm�n  �          AP  �(�@��R��ff��C���(�Az��vff��  C�                                    Bxnm�  �          AQ���@����������Cz���A�H�u����C�q                                    Bxnm��  �          AP�����@�z��	�%Q�C�����A=q��������C�                                     Bxnm�`  �          AO\)�G�@�p��	G��%��C!H�G�A�������C                                      Bxnm�  �          AO����@����\)�"C�{���Az���ff��(�CaH                                    Bxnm׬  �          AMp���@���\)�$��C����A������CxR                                    Bxnm�R  T          AL  �
=@�p���(���CG��
=A
{��(�����CY�                                    Bxnm��  �          AK����
@�������(C�����
AG���(�����C�                                    Bxnn�  �          AIG���ff@�G���
�)�CY���ffA�����\���RC}q                                    BxnnD  �          AJff��Q�@�33��p����C����Q�AG��]p��~�RC�q                                    Bxnn �  �          AN{��p�@�33��:33C�{��p�A{��p�����C޸                                    Bxnn/�  T          AMG��ᙚ@w���
�L�RC@ �ᙚA���ָR��Q�C&f                                    Bxnn>6  �          AO
=��{@^{�ff�IffC�q��{@�
=�ڏ\� =qC�                                    BxnnL�  �          AO�
� ��@.{����DG�C!Y�� ��@�����  C	�                                    Bxnn[�  �          AN�\�   @^{�(��=��C���   @�����\)��33C��                                    Bxnnj(  �          AMp��ڏ\@`���$z��U
=C�\�ڏ\A33���
��HC�                                    Bxnnx�  �          AO\)���H@33�4  �q  C!h����H@��
=�(z�CY�                                    Bxnn�t  �          AV�\�Q녽��N=qu�C6
=�Q�@�z��333�eG�B�B�                                    Bxnn�  �          AXz��l(��B�\�O�
=qC6��l(�@�z��5��c��B�
=                                    Bxnn��  �          AXz��p�׿Q��O��C@Y��p��@��\�9��l�\B���                                    Bxnn�f  �          AW
=�G���p��N�H#�CP��G�@�G��@z��{B�                                    Bxnn�  �          AX(��J�H�����O���CS���J�H@�(��B�H�B��                                    Bxnnв  T          AW\)�QG����R�PQ���C9}q�QG�@�=q�6ff�h�B�Ǯ                                    Bxnn�X  �          AU�)�����N�H��CVǮ�)��@��R�Ap��B�G�                                    Bxnn��  �          AV�H�\)�
=�O�ffC_k��\)@�G��F{��B�                                     Bxnn��  �          A[���z�>�p��F�\#�C0k���z�@Ӆ�((��G��C��                                    BxnoJ  �          A\����Q�>\)�K�\C2s3��Q�@ҏ\�.ff�Q(�C �f                                    Bxno�  �          A]����p��u�J{��C6n��p�@�ff�0  �R�Ch�                                    Bxno(�  �          A[�
���H�xQ��A�xffC<�����H@�=q�.�R�Sp�C�                                    Bxno7<  �          A[33���H�G��D���
C;�H���H@��H�0  �VCG�                                    BxnoE�  �          A[�
�ə�=�Q��AG��y�
C38R�ə�@�
=�%�E�HC\)                                    BxnoT�  �          A[��Å>��
�C
=�}G�C1��Å@�\)�%G��D�HCY�                                    Bxnoc.  �          AY�3�
�   �O�
G�CWu��3�
@����C33Q�B��                                    Bxnoq�  �          AS�
�%�,���K3333Cb+��%@���D���{B��
                                    Bxno�z  �          AT���   �=p��K33\Ce�
�   @{��F�H�fB��)                                    Bxno�   �          AYp�����H���Pz�u�Ck�����@|���L��L�B��f                                    Bxno��  �          A\z���  ��J�\CO�
��  @�p��?33�{
=C�\                                    Bxno�l  �          Aap��ə�>�(��H���}33C0!H�ə�@����)��C�C޸                                    Bxno�  �          Aa��G��   �O��\C9)��G�@ƸR�6ff�WffC�q                                    Bxnoɸ  �          Ab=q��=q?fff�L����C+5���=q@��H�(���A��C h�                                    Bxno�^  �          ALz��g
=>�ff�@Q�Q�C,���g
=@љ��!G��UB��R                                    Bxno�  �          A=�����H�w��/�
�fC�g����H@�
�6�R��B��
                                    Bxno��  �          AG�
@!G���z��&�R�bffC�@!G��\)�C���C�p�                                    BxnpP  T          AEp�@.�R����{�U��C���@.�R�E��?�W
C�"�                                    Bxnp�  �          AG�@���p�����,�RC��R@���#33�7���C�7
                                    Bxnp!�  �          AL  @|���G����3��C�f@|����
�=��ǮC��3                                    Bxnp0B  �          AO
=@0  ���
�{�G\)C��@0  ��z��HQ�.C�o\                                    Bxnp>�  �          AO
=@XQ����=q�@��C��@XQ��=q�E��aHC���                                    BxnpM�  T          AM�@Fff�����\�C��C���@Fff���H�D���)C��=                                    Bxnp\4  �          AH��@7
=��������MC��q@7
=��
=�B�\k�C��3                                    Bxnpj�  �          AK
=@C33���33�N��C�n@C33�����D(�p�C�\)                                    Bxnpy�  �          AHz�@@���أ��!��V��C�  @@�׿���A��\C�j=                                    Bxnp�&  �          AF=q@S�
����33�N�HC��@S�
�fff�>{ǮC�e                                    Bxnp��  �          A<z�@�����ff��Q��C�'�@����K��!�n{C���                                    Bxnp�r  �          A@Q�@\)��p��	���C�C��=@\)��
=�1���C���                                    Bxnp�  T          AC
=��33��{�(���p�RC����33>\�A¨�\C!�H                                    Bxnp¾  �          AB�H��G���ff�+�
�x33C�lͽ�G�?0���B{­k�B��                                    Bxnp�d  �          AC33?��R��33�$���f��C���?��R��Q��A��¥��C�<)                                    Bxnp�
  �          AEG�?�  ��  � ���]{C�AH?�  �\)�Ap�¢C�.                                    Bxnp�  �          AA�?�
=���
���W  C��{?�
=�Q��>=q   C�l�                                    Bxnp�V  �          A@��@G���=q�ff�]  C��=@G�����=� 33C�o\                                    Bxnq�  T          AB�H?���ff� (��]�\C�H?����@z�£k�C�Ff                                    Bxnq�  �          ADQ�?�����p���
�ZG�C��H?��Ϳ8Q��B{£�=C�ٚ                                    Bxnq)H  �          AEG�?�p���z�����V(�C��?�p���  �AG�¥{C�|)                                    Bxnq7�  �          A(z�@�H��{�p��d(�C��H@�H>.{�$  W
@z=q                                    BxnqF�  �          A,��@\)��ff����[(�C���@\)��\)�'�
.C���                                    BxnqU:  �          A2ff@R�\��
=����F�\C�w
@R�\�����*�\\C��R                                    Bxnqc�  �          A1��@/\)��p���H�E�HC�(�@/\)��ff�*�RL�C�XR                                    Bxnqr�  �          A4  @<(���������=�C�Y�@<(���
=�+�
��C�'�                                    Bxnq�,  �          A5@>�R��G����*�C�z�@>�R�(���)��fC�E                                    Bxnq��  �          A6=q@AG���(���33�({C�w
@AG��0  �)���HC��\                                    Bxnq�x  �          A6�R@{�������4z�C��3@{��
�.{� C�~�                                    Bxnq�  �          A8z�@P������
=�=\)C�e@P�׿�{�.=q\C��                                    Bxnq��  �          A7\)@�����ff�33�F�\C�|)@��׿=p��*�\C���                                    Bxnq�j  T          A4��@u���Q�����4
=C�T{@u�����'�
=C�xR                                    Bxnq�  �          A4��@|(�����Ǯ�{C�]q@|(���������pp�C�4{                                    Bxnq�  �          A<z�@q���R��������C�y�@q��������iG�C�˅                                    Bxnq�\  �          AG�@(��5�>=qC�T{@(�@s�
�9ffBez�                                    Bxnr  �          AJ�H?��������G�£�{C�Ǯ?���@�(��2�H�z{B�(�                                    Bxnr�  �          ADz�?�Q쿕�Bff¤W
C��R?�Q�@�(��/33�}=qB�
=                                    Bxnr"N  �          AD��?�\)����7
=\C�.?�\)@33�@Q�B�(�                                    Bxnr0�  T          AG�?�
=�u��<  ��C�4{?�
=@:�H�@(�B�
                                    Bxnr?�  �          AB�H?u�Q��:=qB�C�*=?u@U��9��
B�                                    BxnrN@  �          AG
=?�Q���R�@��G�C��f?�Q�@�
=�8��ffB���                                    Bxnr\�  �          AFff>��
�W��>ff=qC���>��
@X���>=q  B�(�                                    Bxnrk�  �          AL  ?�  ��Q��3��z�C�Ф?�  ?�(��G33 �=B\)                                    Bxnrz2  �          AF�\@�{��
=�G��8�C���@�{���R�4��C���                                    Bxnr��  T          AO\)@�����H��z����HC�Z�@�����33��R�M�C�g�                                    Bxnr�~  �          AO�A
=������R���C�p�A
=��G�����ޣ�C��q                                    Bxnr�$  �          AMG�A(���?0��@Dz�C�+�A(��
{�������RC��                                    Bxnr��  �          AK33@�\��@�\A(��C�K�@�\����P���r{C���                                    Bxnr�p  �          AIG�@�����@�  A�
=C�q�@��� �ͽu��=qC�7
                                    Bxnr�  �          AK\)A(����?�33@�{C�k�A(�����=q���
C���                                    Bxnr�  �          A4��@�33�=q@E�A}��C��)@�33�  �������C��                                    Bxnr�b  T          A1@�(��
{@p  A�(�C�&f@�(���ÿ�\)��C�&f                                    Bxnr�  T          A5��@�  ��
@��A��C�:�@�  ��ÿ���{C��                                    Bxns�  �          A:{@�=q��@���A��HC��)@�=q� �þ�33��(�C�f                                    BxnsT  �          AF�H@��H��
@�33A��HC�U�@��H�,  �0���Mp�C��H                                    Bxns)�  �          AJ�H@ƸR�=q@���A�z�C�b�@ƸR�/��(��0��C��)                                    Bxns8�  �          AU�@��  @���A�(�C�0�@��/�
�L���^�RC��{                                    BxnsGF  T          AY�@�p���@�RB  C��@�p��6�H?�33@޸RC��                                    BxnsU�  �          A\��@�(���@�(�B\)C���@�(��?�
?�  @ǮC�y�                                    Bxnsd�  �          AX��@�{���@��B�C��R@�{�<z�?�\@�C�W
                                    Bxnss8  |          A`��@�\)�Q�Ap�B�C�]q@�\)�A�@�A�C��3                                    Bxns��  �          Af�R@�ff�Q�A�B��C��@�ff�D��@
�HA
�HC�.                                    Bxns��  �          Aa��@�  ��A��B�C���@�  �@��@#33A'
=C���                                    Bxns�*  �          A`��@�=q�
{A�RBC���@�=q�@��@>{AC�C�K�                                    Bxns��  �          A_�@����	p�A�B��C���@����?�
@<(�AB�RC�J=                                    Bxns�v  �          A\��@��
�	A\)Bp�C�AH@��
�>�R@333A;33C��                                    Bxns�  �          A[33@����A�RB#  C�>�@���<��@HQ�AS�
C���                                    Bxns��  �          A[
=@����G�A�
B)�HC�k�@����<  @aG�An�HC���                                    Bxns�h  T          A\��@�������Ap�B6{C��{@����;\)@�Q�A��C�`                                     Bxns�  T          A[�@��H����A#\)B@33C��@��H�9��@���A���C�޸                                    Bxnt�  �          AXz�@����G�A�HBp�C���@����5G�@5�AB�RC�4{                                    BxntZ  �          A[33@���@��B(�C���@��7�?�=q@�ffC�!H                                    Bxnt#   �          AXQ�@�G���\A z�BffC�s3@�G��6ff@�
A{C�c�                                    Bxnt1�  �          AS�@����
=A+\)BY  C�q@���/�@�  A�z�C�Ff                                    Bxnt@L  �          AS
=@����A(��BVQ�C��q@���-��@�(�A�p�C��=                                    BxntN�  �          ANff@��ָRA
=BCG�C���@��-��@��A�  C�                                    Bxnt]�  �          AJ�R@�
=���RA+�Bhp�C�f@�
=��R@�A���C�"�                                    Bxntl>  �          AI�@��
���RA2�HBx�HC�*=@��
�z�@�ffB
=C��                                    Bxntz�  �          AJff@����ffA\)BP�C�Ff@���"=q@��
AɮC���                                    Bxnt��  �          AL  @�����z�A�B=��C��@����&�\@�Q�A�  C��f                                    Bxnt�0  �          AM��@����ϮAG�B:�C�l�@����'�@�p�A���C���                                    Bxnt��  T          AO
=@����p�A�BC�C���@���*{@���A�  C���                                    Bxnt�|  �          AP��@��\��(�A ��BI�RC��@��\�,(�@��HA��C�*=                                    Bxnt�"  �          AC�
A Q����@�G�AǙ�C��fA Q��p�>�p�?�\C�)                                    Bxnt��  �          A1G�@�G��޸R@n�RA�\)C�q@�G���(����;�C�W
                                    Bxnt�n  �          A0(�@����ff@���A���C���@�����>��?�\)C�:�                                    Bxnt�  �          AG�@Ǯ��@��HBG�C�}q@Ǯ��ff?���AC��q                                    Bxnt��  �          A�@�  ����@�z�B�RC���@�  ��\@(Q�Ax(�C���                                    Bxnu`  T          @Å@S33�|(����H��p�C��)@S33�Dz��!G����
C��\                                    Bxnu  �          @�ff@g
=��\�H����HC��R@g
=>B�\�fff�3\)@@                                      Bxnu*�  �          @�(��W
=��(�@;�B�ffCt�;W
=��Q�@\)BD�C���                                    Bxnu9R  �          @��
=q�h��@�G�B��CJ���
=q�w�@���B3{Cp�)                                    BxnuG�  �          @�(�?�(�?���?�z�A��BK��?�(�?h��?��HBA��A�=q                                    BxnuV�  �          @��R�#�
@\)@$z�B=qC
�q�#�
?�@W�BQ{C(�
                                    BxnueD  �          @�{��
=�s33@u�B}G�CQuÿ�
=�6ff@.{B�HCo��                                    Bxnus�  �          @�����H��=q@`  Bz
=CX�����H�2�\@�B�
CrT{                                    Bxnu��  �          @��H�����\)@g
=B�B�C7�f����   @@��BS��CqL�                                    Bxnu�6  �          @5��?z�@�B�k�C{���.{@
=qB�� Ch=q                                    Bxnu��  �          @?\)��
��p��������C=33��
>�(���\)��ffC)u�                                    Bxnu��  �          @{���33�k��:�H�q(�C;����33?�\)�%��I(�C8R                                    Bxnu�(  �          @{���{�����@  �w��C����{>�z��W�¤C��                                    Bxnu��  �          @�\)�L�Ϳ���@h��B�{CqT{�L���K�@�\BC��                                    Bxnu�t  �          @z=q���R��ff?c�
A�{Ckc׿��R�   �W
=����Cn&f                                    Bxnu�  �          @�Q��  ��{@Q�Bu�\C>�R��  �33@%�B/  Ceh�                                    Bxnu��  �          @w
=�[�����?��RA���C8���[��k�?c�
A`z�CB�q                                    Bxnvf  �          @o\)�P  �8Q�?��A�=qC7&f�P  �z�H?���A��
CD��                                    Bxnv  �          @S�
�!논��
?�Q�A�{C4ff�!녿J=q?��HA�Q�CEff                                    Bxnv#�  �          @Fff����aG�?�z�B�z�C[�׾���L��?\(�B5�HC|(�                                    Bxnv2X  �          @j�H���R>\)@E�B�Q�C.n���R���@+�BM��Ca�3                                    Bxnv@�  �          @�(��8Q쿀  @�z�B�W
Cjc׿8Q�����@�z�B?�C��                                    BxnvO�  �          @�{@
=q��  @�ffB2��C�t{@
=q�Å?�
=Ah��C��
                                    Bxnv^J  �          @��
@)����(�@��A���C��q@)�������0����p�C�Ф                                    Bxnvl�  �          @�
=@A����@�33B\)C�K�@A���ff?fff@�
=C�                                    Bxnv{�  �          @�p�@�����@��B*{C��q@�����?�  AP��C�)                                    Bxnv�<  �          @��@p����@�B4�C�@p���\)?�\)Ay�C��f                                    Bxnv��  �          @��?����w
=@��BMQ�C�ff?�����p�@�RA��C�Ǯ                                    Bxnv��  �          @ƸR?Y���+�@�33By�C�Ф?Y����z�@Dz�A��
C��{                                    Bxnv�.  �          @�
=?�{�@  @��Bq��C�!H?�{��  @[�A��C���                                    Bxnv��  �          @�ff@(���Tz�@��
B](�C�7
@(����33@W�A��C���                                    Bxnv�z  �          @�@���p�@�{BvC�|)@����G�@��B�C��                                    Bxnv�   �          @�(�@���%@љ�B{Q�C���@����
=@��
B(�C�C�                                    Bxnv��  �          @�
=@�
���\@6ffA��C���@�
��=u?+�C�Ф                                    Bxnv�l  �          @��@333�4z�@�ffBLC�b�@333��G�@(��A���C��f                                    Bxnw  �          @��@�
�;�@���B<��C�!H@�
����?���A�ffC�=q                                    Bxnw�  �          @�Q�@)���r�\@?\)B�\C�t{@)������>��@�ffC�k�                                    Bxnw+^  �          @��
@O\)����@Q�A�Q�C���@O\)���;�G����RC�t{                                    Bxnw:  T          @�Q�@���Z=q?��A�
=C�'�@���xQ쾀  ��RC�Q�                                    BxnwH�  �          @���@u��|��@=p�A��C��@u���p�>�Q�@Q�C��                                    BxnwWP  �          @���@����@��B$G�C��3@���r�\@!G�A�33C���                                    Bxnwe�  �          @��\@?\)�\)@���BF�\C���@?\)��Q�@   A��HC���                                    Bxnwt�  �          @i��?��H�\@\)B4C�)?��H�'�?��
A��C�^�                                    Bxnw�B  �          @l(�?�\)�k�@5BaC��?�\)��p�@G�B&�C��=                                    Bxnw��  �          @���?���:=q@��BR�C���?�����
@p�A���C��                                    Bxnw��  �          @ٙ�@G��b�\@���BM�RC�U�@G����\@!�A�Q�C��                                    Bxnw�4  �          @�\)@E��g
=@��B<��C�9�@E���G�@
=A�=qC���                                    Bxnw��  �          @�Q�@���:�H@��B0��C�!H@�����\@ ��A�
=C�Q�                                    Bxnẁ  �          @�(�@<���7�@�p�BM��C���@<�����R@3�
A��C��f                                    Bxnw�&  �          @���@=q�}p�@�=qB2z�C���@=q����?��HAtQ�C�H�                                    Bxnw��  �          @�  ?�p�����@���B7��C��?�p����?�p�Axz�C�C�                                    Bxnw�r  �          @�  @�\�g�@���BN�HC���@�\��z�@�RA�C���                                    Bxnx  �          @�?��R�l��@���B[
=C���?��R���@0��A�(�C��                                    Bxnx�  �          @��
?����r�\@��B[p�C��?�����
=@+�A��
C�                                      Bxnx$d  �          @�(�?#�
�~�R@�G�BWC��?#�
���H@!G�A�Q�C��                                     Bxnx3
  �          @�ff?����`��@��BT�C���?�����33@'�A��C�8R                                    BxnxA�  �          @�  ?��
�R�\@���B[{C�5�?��
��(�@,(�A�C�Ǯ                                    BxnxPV  �          @��@ff�K�@�z�BU��C���@ff��ff@'�A���C���                                    Bxnx^�  �          @���?�
=�>�R@��B^  C�u�?�
=��G�@/\)AυC��                                    Bxnxm�  �          @��@vff����@�  B2G�C��{@vff�{�@!�A�\)C�0�                                    Bxnx|H  �          @��@��
��p�@|(�B({C�R@��
�XQ�@!G�Aʣ�C�S3                                    Bxnx��  �          @�=q@��;�33?�z�A��C���@��Ϳ�ff?�  AG
=C�C�                                    Bxnx��  �          @���@�\)����@�\A�p�C��\@�\)��Q�?�  A���C���                                    Bxnx�:  �          @Ϯ@�G��.{@5AУ�C�*=@�G���@33A��HC��3                                    Bxnx��  �          @�G�@�G�?�Q��<����p�A�@�G�@&ff���c\)A��                                    Bxnxņ  �          @�{@�\)?p���\(��ָR@��@�\)@)�������{A��                                    Bxnx�,  �          @��@��?��}p�����@�
=@��@%��C33����A�=q                                    Bxnx��  �          @�ff@��>���(���  @��@��@(Q��N{��A�z�                                    Bxnx�x  �          A ��@�ff��{�l(���  C�w
@�ff?W
=�u���
@�p�                                    Bxny   �          @�
=@�
=?���=q� G�@�{@�
=@(���H�����A���                                    Bxny�  �          @�@�33?xQ���(���Ap�@�33@K��J=q��p�A���                                    Bxnyj  �          @�\)@��׾8Q��xQ����C��@���?�Q��W���p�A���                                    Bxny,  �          @�  @�=q��p��HQ��֏\C�H�@�=q?&ff�S33��
=@���                                    Bxny:�  �          @��@ȣ׿�녿�\�qp�C���@ȣ׿#�
�   ��G�C��                                    BxnyI\  �          @��@�G��5�����(�C��)@�G���
��33�]��C��=                                    BxnyX  �          @�\)@ȣ��<��?�@�G�C�c�@ȣ��1G���33���C�
                                    Bxnyf�  �          @��@���c33@k�A�
=C�XR@����\)?�
=A�\C���                                    BxnyuN  �          @�@e�\��@���BE�C��@e����@A�A���C�`                                     Bxny��  	�          @���@*�H�S33@���Be��C�z�@*�H�˅@u�A�(�C�^�                                    Bxny��  
�          A  ?�{�"�\@��B��qC�3?�{�ȣ�@��\B�\C�                                      Bxny�@  T          @�p�?�R��=q@�p�B��
C�q�?�R����@���B-��C�8R                                    Bxny��  
�          @�\��p�=�G�@�\B�W
C0^���p��w
=@ǮB_p�Cu�=                                    Bxny��  
�          A zΉ��Ǯ@�B�W
CG�=�����33@�
=BWp�C�N                                    Bxny�2  T          A�H��{��z�Az�B�8RC�'���{��p�@�z�B@��C�W
                                    Bxny��  
�          @�(�����@1G�����W
B�LͿ���@�Q���z����B�p�                                    Bxny�~  
�          @��H@@���8������2\)C��@@�׾aG������k�HC��                                    Bxny�$  T          @�Q�@g
=�Dz��b�\��C��3@g
=�.{��(��N=qC���                                    Bxnz�  �          @�ff@p��*=q��(��HG�C��R@p����
��p���C��H                                    Bxnzp  �          @�G�?��
���H�����C���?��
?�=q���
��B�H                                    Bxnz%  "          @��
@�
��G����v�C�4{@�
?�����fffB��                                    Bxnz3�  �          @���@!G��=p���33�q=qC��@!G�?��������WffBff                                    BxnzBb  
�          @���?��
�\��z��C��)?��
@�\�z=q�\��BC�\                                    BxnzQ  "          @n{����?\�C33�h�RB��ῐ��@:�H��  ���
B��                                    Bxnz_�  �          @c33�#�
?����L���fB�.�#�
@*�H��(�B��                                    BxnznT  �          @����W
=?������R�RB�\)�W
=@_\)�1���HB��
                                    Bxnz|�  T          @��
��ff@
=q�j�H�c  B�Ǯ��ff@r�\��
=���HB��                                    Bxnz��  
(          @����s33>\)��(���C+�Ϳs33@/\)��G��YG�B��                                    Bxnz�F  �          @E�\(���
=���IQ�Cr��\(��L���5�ffC@�\                                    Bxnz��  �          @�33�
=q�U�L���5p�Cm{�
=q����{���Ccٚ                                    Bxnz��  T          @8Q��\���
?��A��C\�H��\� ��>��R@ҏ\Cd�                                    Bxnz�8  
�          @E�{��Q�?���A�RCT���{���R>�A
�HC]�R                                    Bxnz��  �          @�(��g���(�@�RB�HCF�H�g��
=?�Q�A���CU�                                    Bxnz�  T          @��a녿��R@��B(�CGQ��a��
=?�z�A��RCU�=                                    Bxnz�*  
�          @z=q�.{���@p�B&C=�q�.{���H?���A�z�CT33                                    Bxn{ �  �          @��R�(����@Q�B
=CY�)�(��.�R?J=qAD  Cd@                                     Bxn{v  �          @���(��L��?�\)A��RCh��(��\(��\)��G�Cj��                                    Bxn{  T          @��H>aG������\��G�C�\)>aG��hQ��0����RC��3                                    Bxn{,�  
�          @��?^�R��z�z�H�*�\C��{?^�R�s�
�e��*  C�u�                                    Bxn{;h  T          @�
=?�33��\)�U���(�C�!H?�33�k���G��eQ�C��                                    Bxn{J  
Z          @�p��W
=��녿���j�\C�K��W
=�c33�vff�9�C�Z�                                    Bxn{X�  
�          @ƸR?���������G����C�j=?����|�����H�?  C�<)                                    Bxn{gZ  
�          @�\@���˅�S33��p�C�P�@���fff��{�_��C�aH                                    