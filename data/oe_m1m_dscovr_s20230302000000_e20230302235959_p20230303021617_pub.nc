CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230302000000_e20230302235959_p20230303021617_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-03T02:16:17.968Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-02T00:00:00.000Z   time_coverage_end         2023-03-02T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxi��  �          @�  ��?�33@�z�B���C)������@���B�\Cc�{                                    Bxi��&  �          @�\)���
@�H@�(�Bx�B��ᾣ�
��33@��RB��{Cd                                    Bxi��  
�          @����G�@�@��
Bx�B�.��G���{@��RB�8RC{޸                                    Bxi�r  �          @�  ���@{@��
Bv�
B�\�������@�\)B�Q�Cd��                                    Bxi�   T          @�
=���@,��@��RBk�B�uþ�����
@��RB���C<G�                                    Bxi�.�  
�          @�>.{@)��@�{Bl�B��>.{�L��@�p�B���C�Q�                                    Bxi�=d  �          @��?(��@Fff@�Q�BUffB�#�?(��>�(�@�  B�\B�\                                    Bxi�L
  �          @���?J=q@J=q@�p�BP
=B�{?J=q?�@��RB�k�B33                                    Bxi�Z�  �          @���?G�@K�@�p�BOp�B�L�?G�?
=q@�
=B�G�B
�
                                    Bxi�iV  �          @���?Q�@C�
@���BR��B�\?Q�>�ff@�z�B�aHA�
=                                    Bxi�w�  �          @�{?�\)@���@#33A�G�B��?�\)@
=@���Bs=qBx=q                                    Bxi���  �          @�33?У�@�G�?�z�A��
B��\?У�@@��@q�B?=qBv�\                                    Bxi��H  
�          @�p�?�p�@�  @=p�B
=B���?�p�?��@�33B�
=Bc                                      Bxi���  �          @�=q>��?�@�ffB�L�B�ff>����33@�33B�  C�XR                                    Bxi���  �          @��?��
@�=q?�z�AH��B���?��
@n�R@hQ�B*�\B�                                    Bxi��:  �          @��H?�{@�G�@>{B�\B��
?�{@�@���B}p�Bx�                                    Bxi���  �          @�33?��R@\)@U�B33B�W
?��R?�
=@�p�B��BU                                    Bxi�ކ  
�          @�33>�=q@���@L(�BB��
>�=q@   @�ffB��RB�                                    Bxi��,  �          @��H=L��@qG�@p��B3�\B�\)=L��?��\@�p�B�L�B�(�                                    Bxi���  T          @���>��@tz�@s33B3G�B�z�>��?��@�\)B�
=B��                                    Bxi�
x  
�          @�p�?(�@%@�
=Br(�B�z�?(��k�@�(�B�(�C��{                                    Bxi�  �          @���=��
?�p�@�33B�p�B�u�=��
��z�@�  B���C�                                    Bxi�'�  
�          @��L��?�@�=qB���B���L�Ϳ��@�G�B�  C�Ǯ                                    Bxi�6j  �          @�
=��\)@   @��\B�u�B�{��\)��ff@�33B�  C~�                                    Bxi�E  T          @���    @   @�G�B~�B�33    ��@��B�=qC��)                                    Bxi�S�  �          @���.{@��@�{B�#�B��)�.{�!G�@�B��CG�                                    Bxi�b\  �          @��R��R@�@�\)B�L�B�ff��R�fff@�=qB��Ck�=                                    Bxi�q  
�          @���\(�@��@�{B��
B��f�\(��+�@��B��CY�f                                    Bxi��  �          @��H�(��?�Q�@�B��fB�aH�(�ÿ��\@�{B���CmO\                                    Bxi��N  T          @�����R?�@�(�B��B�  ���R����@��HB�ffC}�3                                    Bxi���  
�          @��R��=q?�p�@��RB��qB�zᾊ=q��@�\)B�C���                                    Bxi���  
Z          @�  �Ǯ@'
=@��\BuffB�\�Ǯ���@�\)B�.CU                                    Bxi��@  �          @���>�=q@J�H@��BZ�B�ff>�=q>���@�\)B�
=B`�                                    Bxi���  �          @�p���\)�   @H��B:(�Cm���\)�p��?�33A�
=Cv�                                     Bxi�׌  �          @������H�z�@`  BJG�Ci�=���H�r�\?�ffA���Cu��                                    Bxi��2  T          @��ÿ�G��z�@Z=qBT\)Ci����G��a�?���A�(�Cv޸                                    Bxi���  �          @��
��33��z�@W
=B[�\Ci��33�W�?�33A�  Cwu�                                    Bxi�~  "          @j=q��(��Y��@Tz�B���CV�=��(��(�@=qB%�\CsW
                                    Bxi�$  �          @q녿�G��Q�@c33B��=C[
��G��!�@'
=B/�Cx.                                    Bxi� �  T          @{���=q�s33@h��B��fC]^���=q�,(�@(Q�B(��Cx#�                                    Bxi�/p  
Z          @�=q��(����
@�  B��\C\{��(��<(�@9��B)p�Cwu�                                    Bxi�>  T          @�ff��ff�\)@�=qB�33C;����ff�&ff@��B]�Cx�                                    Bxi�L�  "          @��ÿTz�?G�@�z�B��qC��Tz����@�33B�z�Cu�                                     Bxi�[b  
�          @��\�z�?�
=@�z�B���B�G��z��  @���B�Cx�=                                    Bxi�j  
�          @������@@��HB���B�k�����\)@�Q�B��
Ct�{                                    Bxi�x�  
�          @�p�����@
=q@�B�\Bè��������@��
B�{Cq��                                    Bxi��T  �          @�=q>��?�  @��B���B��)>���n{@��RB�
=C���                                    Bxi���  R          @�z�=L��?���@�33B�.B��=L�Ϳ��@�
=B�
=C�T{                                    Bxi���  
(          @��׾���?��
@�(�B�Q�B�33������  @�{B��\C�%                                    Bxi��F  �          @�G��W
=?���@��\B�k�B�G��W
=��(�@�G�B�=qC��q                                    Bxi���  
�          @��;��?�  @�
=B�B�B����׿���@��
B���C}�H                                    Bxi�В  V          @�=q���?z�H@�ffB�B�aH��Ϳ���@�{B�L�C~Y�                                    Bxi��8  �          @�Q�(��?�{@�33B��
B��(�ÿ��
@�p�B���Cy�                                    Bxi���  �          @�33�^�R?#�
@�\)B���C�׿^�R�  @���B��3Cx��                                    Bxi���  T          @��R�k�?�z�@�
=B�
=B��;k���\)@���B���C�AH                                    Bxi�*  �          @����  ?�ff@��B�� B���  ��Q�@���B�33C�3                                    Bxi��  
�          @���8Q�@�R@���B��B�p��8Q�5@�=qB�CaH                                    Bxi�(v  �          @��\��=q@   @�
=B|��B����=q�Ǯ@���B�\)Ck�\                                    Bxi�7  �          @����\)@!G�@�p�B{z�B�33�\)��{@���B�Q�Cxh�                                    Bxi�E�  T          @�=q�8Q�@p�@�=qB{p�B�Q�8Q쾮{@���B��Cq��                                    Bxi�Th  �          @�
=��(�>�\)@��HB�.C)p���(��
=@���Bi��Cm�q                                    Bxi�c  �          @��
���H@�H@p��Bb�B�
=���H>.{@�
=B���C �R                                    Bxi�q�  �          @��R?��R@��H@5A��B�  ?��R@�\@���Bn=qBc�                                    Bxi��Z  T          @�(�?��H@Q�@�ffBU�B���?��H>�(�@�
=B�A��                                    Bxi��   W          @�=q?�  @w�@r�\B,  B��?�  ?�33@�\)B�p�BA
=                                    Bxi���            @���?��@dz�@�z�B<�RB�u�?��?z�H@�(�B���B{                                    Bxi��L  �          @���?�Q�@S33@��BF��B�Ǯ?�Q�?0��@��\B���A˙�                                    Bxi���  T          @��?���@S33@��RBH�B���?���?5@���B��=A�                                    Bxi�ɘ  
�          @�=q�W
=@=p�@�
=BWp�B�uÿW
=>�
=@�z�B�k�C�=                                    Bxi��>  �          @��\��ff@2�\@���BT��B�.��ff>u@��HB��RC++�                                    Bxi���  �          @�G���
=@.{@���BY�HB���
=>#�
@��\B�aHC-xR                                    Bxi���  
�          @��\���R@+�@��B`�
B�z῞�R=��
@��B�C0aH                                    Bxi�0  �          @�ff��(�@0  @�Bm33B���(�    @�B��)C4
                                    Bxi��  �          @��R�Y��@�@��HBy�B�\)�Y������@��
B�33CM(�                                    Bxi�!|  T          @�녿�  @{@�\)Bj(�B�p���  ���@��\B�#�C=�f                                    Bxi�0"  T          @��ÿ��@ ��@�
=B�B�LͿ���G�@��B�W
CS0�                                    Bxi�>�  �          @�\)��
=?ٙ�@�=qB���B��쿗
=����@��B��qC^��                                    Bxi�Mn  �          @�  ���H?�=q@��
B��fC	�����H��Q�@��HB�.C`��                                    Bxi�\  �          @����(�?�G�@�33B�=qC�
��(��ٙ�@��B33C`�
                                    Bxi�j�  T          @��ÿ��H>�{@���B��3C*Ϳ��H�33@��
BcQ�Ce�                                     Bxi�y`  	�          @�\)�
=��{@���B��C=��
=�5�@�BGQ�Ci8R                                    Bxi��  �          @�(��,�Ϳ8Q�@��\Bj��CBٚ�,���;�@fffB(��CcO\                                    Bxi���  T          @�z��*=q���@a�BG��CN���*=q�;�@Q�A�ffCc��                                    Bxi��R  "          @���0�׿�G�@Dz�B,p�CT� �0���C�
?�\A��Cc�3                                    Bxi���  "          @�p��,(��   @7�B"G�CX�)�,(��J�H?�p�A���Ce��                                    Bxi�  �          @p���{�^�R?��B{CT�
��{��z�?333A�ffCb                                      Bxi��D  �          @%��O\)�L��?У�B{
=CA�)�O\)�}p�?��B6��CfxR                                    Bxi���  �          @e?�=q?�Q�?�=qB6(�BT��?�=q>���@33B�
A���                                    Bxi��  �          @��\���
���@\��BwQ�CVc׿��
�(Q�@{B{Co��                                    Bxi��6  "          @r�\��zῦff@<(�BOG�CVB���z��&ff?�33A�\)Ci��                                    Bxi��  T          @X���   ���R@  B(�CX��   �=q?��HA���Cfn                                    Bxi��  
�          @}p��\)��\@��B"p�CZ@ �\)�0  ?��RA��Cf�)                                    Bxi�)(  
Z          @����z�!G�@|(�B�8RCL���z�� ��@FffB<ffCp�3                                    Bxi�7�  T          @�\)��z�(�@|(�B��=CKW
��z��\)@G
=B=��Cp��                                    Bxi�Ft  
�          @\)���=�Q�@q�B���C.�q�����ff@Tz�Bgz�Co�                                    Bxi�U  �          @x��@z�@'�?��A��BA�H@z�?���@.�RB3�B =q                                    Bxi�c�  �          @��@z�@(��@p�B ��BB�\@z�?�Q�@N�RBLG�AظR                                    Bxi�rf  T          @�@%@*=q?�z�A�B7  @%?�{@>�RB6(�A��                                    Bxi��  �          @�Q�@	��@)��@"�\B�\BK��@	��?��@aG�B_\)A�                                    Bxi���  �          @�p�@
=q@'�@5�BffBJ(�@
=q?^�R@p��Bi�A��                                    Bxi��X  �          @�z�@�R@6ff@{B	Q�BO��@�R?�  @c�
BY33A�G�                                    Bxi���  �          @��@)��@5?�(�A�p�B<(�@)��?��R@G�B733A��H                                    Bxi���  T          @�ff@�@7
=?�
=A�=qBI�@�?\@FffB>�B(�                                    Bxi��J  �          @���@33@/\)@
=A�(�BHG�@33?��@L(�BH�
A��                                    Bxi���  "          @��@!G�@0��?�G�A��
B>��@!G�?��
@9��B2�A�(�                                    Bxi��  �          @�\)@�@=p�?��
A�BJ�@�?�Q�@@��B5�B
�                                    Bxi��<  �          @�z�@(Q�@8��?��A�{B?
=@(Q�?�\)@$z�B�B�                                    Bxj �  "          @Y��?�33?�33?�A�Q�B4�?�33?Tz�@��BD�A��                                    Bxj �  �          @R�\?�p�?�(�?��HB{BB��?�p�?@  @+�B[G�A�ff                                    Bxj ".  T          @HQ�@�
?�
=?
=qA#�B�@�
?�z�?�A�  A��
                                    Bxj 0�  "          @>{@ ��?�
=?�Q�B�\A�@ ��>L��@33B5@��
                                    Bxj ?z  �          @3�
?
=q=�G�@\)B��)A=�?
=q��\)@{Br��C��                                    Bxj N   �          @@�׾\�0��@8Q�B��=Cp�)�\� ��@
�HB:�C��H                                    Bxj \�  �          @b�\�c�
���@U�B���C=p��c�
��\@5�B\Q�CsT{                                    Bxj kl  �          @R�\�Q���
?���A�RC[ٚ�Q��z�?   AQ�Ccz�                                    Bxj z  �          @B�\���Ϳ�=q?�p�B��C\^������{?=p�Adz�Cf+�                                    Bxj ��  �          @C33�%��\(�?G�A��RCFW
�%�����>��@��CK�f                                    Bxj �^  T          @N{�H��?
=q>.{@@  C*@ �H��>Ǯ>��@�(�C,�                                    Bxj �  �          @X���
=�}p�@33B2�CM{�
=���R?��
A�{C_J=                                    Bxj ��  T          @\(��
�H�s33@�HB6p�CK���
�H� ��?��A�\C^��                                    Bxj �P  �          @h���8Q쿪=q?޸RA�ffCL�)�8Q�� ��?^�RA^�RCW                                      Bxj ��  �          @l(��6ff��{?��A�=qCMp��6ff��
?fffAe��CW�\                                    Bxj ��  T          @g
=�C�
�ٙ�?L��ANffCQ��C�
��׾���
CS��                                    Bxj �B  T          @����L(���?���A��HCQ���L(��
=?�@�RCXz�                                    Bxj ��  T          @~�R�HQ��33?��A�
=CO� �HQ���
?E�A1�CXs3                                    Bxj�  �          @[�������@�\B�CP.�����?�z�A�\)C]h�                                    Bxj4  �          @I����\�B�\@\)B7�CHn��\��  ?˅A�
=C\�3                                    Bxj)�  �          @E��{�.{@�
BE��CH��{���H?�Q�BQ�C^��                                    Bxj8�  T          @B�\�����B�\@"�\Bd�CO�Ϳ������?���BQ�Ch�                                    BxjG&  �          @W
=��(����@<��Bu�C]
��(��
=@z�B�\Cr�                                    BxjU�  �          @L�Ϳn{�(�@=p�B��CUJ=�n{��(�@33B:G�Ct�                                     Bxjdr  �          ?�?�R>�{?E�B>=qA��H?�R���
?W
=BT�C�U�                                    Bxjs  T          @	��?�=q?��aG���=qB\)?�=q?����L�����
B*=q                                    Bxj��  �          @S33@!�?�������p�B��@!�@ �׾8Q��O\)B��                                    Bxj�d  �          @c33@0��@
=q�Ǯ�˅B33@0��@�?333A7
=B
=                                    Bxj�
  T          @Z�H@*�H?�=q�(���8z�B	�@*�H?�>���@�  B�
                                    Bxj��  
�          @dz�@>�R?��H��  ��
=AЏ\@>�R?�׾Ǯ���
B �                                    Bxj�V  �          @Q�@3�
?�(���\)��=qA��@3�
?�{�Ǯ��p�A�                                      Bxj��  �          @fff@I��?�=q������A��
@I��?��ÿ(���\A�(�                                    Bxj٢  T          @<(�@*�H?zῆff���ADQ�@*�H?�G��&ff�Lz�A���                                    Bxj�H  
�          ?�Q�?��
>���ff�[�AiG�?��
?�R�.{��
=A�                                      Bxj��  �          @��?��R=��5��  @Z�H?��R>�G��\)�v�HAHz�                                    Bxj�  �          @ff?�z�Ǯ�@  ���C�@ ?�z�<��
�W
=��{>��                                    Bxj:  �          @=q@
=q���R�xQ�����C���@
=q>B�\��  ��p�@�                                      Bxj"�  W          @�@(��E��(��mp�C�B�@(��Ǯ�h����(�C���                                    Bxj1�  �          @E�@7��z�^�R���RC�H�@7���Q쿅����
C��                                    Bxj@,  �          @:�H@%��@  ��=q����C��H@%��\)������C�e                                    BxjN�  "          @.�R@p��:�H�Tz����C���@p���  ��=q���HC��                                    Bxj]x  �          @,(�?�=q�xQ쿬����C�  ?�=q�W
=��33�'�RC��3                                    Bxjl  �          @�R?5��
�������C���?5�������  C�~�                                    Bxjz�  �          @/\)?�=q�����˅C���?�=q��(���p���
=C��                                    Bxj�j  T          @\(��L���Tz�c�
�n�HC�N�L���   �ff�,�C��H                                    Bxj�  �          @]p��L���Z�H��33����C�O\�L���6ff����C��R                                    Bxj��  �          @S33>�  �P�׾�����\C�:�>�  �0  ��\��HC���                                    Bxj�\  
�          @
=<��
����
=�.=qC�G�<��
��(���33�ffC�Y�                                    Bxj�  �          ?�33�L�Ϳ�{��(��r�HCo���L�Ϳ}p����\�Q�Cg8R                                    BxjҨ  T          ?�녿fff��\)=���@;�CqͿfff��p��.{��Q�Cn�                                    Bxj�N  T          ?�\)�#�
��(�>k�@�p�Cy�Ϳ#�
�У׿(���33CxxR                                    Bxj��  
�          @!녿��녽�Q�� ��Cj�ÿ���  ���
�ĸRCf�H                                    Bxj��  T          @k��?\)��Q�����
=CI�3�?\)�W
=����33C7��                                    Bxj@  �          @���:=q���?8Q�A*=qC\
�:=q�   �����RC\��                                    Bxj�  T          @�����'
=@333BCd!H��g�?�  A�Cm#�                                    Bxj*�  �          @�=q��녿�33@S33BZ
=Ci�R����N{@�A��Cv�H                                    Bxj92  T          @n{��\�   @#�
B=�HC@J=��\���@�B�HCW�\                                    BxjG�  �          @ff��>.{?�ffB=�RC-W
���z�?���B-CJB�                                    BxjV~  "          ?�
=�:�H�W
=?���Ba�CC�q�:�H�5?h��B&C`O\                                    Bxje$  "          ?�=q���8Q�?h��Bi�\CF�ÿ��z�?8Q�B)Cc��                                    Bxjs�  �          ?����(���z�?�33B�#�CV��(��J=q?aG�B0��Cqk�                                    Bxj�p  T          ?�z�>�������?��B��HC��\>�������?�
=B:Q�C��                                    Bxj�  �          ?��>��þ�Q�?�Q�B���C��=>��ÿ^�R?c�
B.��C�u�                                    Bxj��  T          @`  ?޸R����@@  Bo�RC�W
?޸R��p�@'
=BC\)C��)                                    Bxj�b  T          @j=q@-p�=L��@ ��B+{?���@-p���\)@  B��C�˅                                    Bxj�  "          @HQ�?��
�aG�@1G�B�C��)?��
�z�@33B&�C�Ff                                    Bxjˮ  
�          @��>�(��Q�@
�HB��HC��R>�(���p�?��B#�C��R                                    Bxj�T  
�          @P  ?��׿��\@0��BiG�C��?������?�{B  C��                                    Bxj��  
�          @U?�{��z�@\)BB�\C�  ?�{��?�A�(�C�9�                                    Bxj��  
�          @U?�z�W
=@'
=BM�C�,�?�z����?�
=B(�C�:�                                    BxjF  T          @[�?��h��@+�BNffC�XR?���\?��HB�HC��H                                    Bxj�  "          @U�?�33�5@1G�Bd
=C�n?�33��33@Q�B!  C��                                     Bxj#�  T          @E?=p����@0��B��C�k�?=p��p�?��HB (�C�@                                     Bxj28  
�          @J=q?:�H��ff@8Q�B�C�ff?:�H�  @z�B$�C���                                    Bxj@�  �          @P��?#�
�J=q@E�B��{C�}q?#�
��@Q�B<z�C�e                                    BxjO�  �          @i���&ff�^�R@]p�B�.CiB��&ff�
=@+�B>\)C~�)                                    Bxj^*  �          @s�
�s33�J=q@fffB�\C[��s33�ff@5B@�HCx                                      Bxjl�  �          @\)�Q녿B�\@tz�B��C^Ǯ�Q���H@C33BHffC{T{                                    Bxj{v  �          @��\�G��s33@xQ�B��qCfz�G��'�@AG�B?ffC}O\                                    Bxj�  T          @��׿�R�p��@u�B�33Cl�{��R�%�@>�RBA�C�>�                                    Bxj��  �          @���(��W
=@�Q�B�.Cj{�(��$z�@L(�BI��C�aH                                    Bxj�h  T          @��\�c�
��{@x��B�ffCq
=�c�
�L��@/\)B�RC~u�                                    Bxj�  T          @�=q�^�R����@���B�� Cl}q�^�R�G
=@Mp�B3G�C~aH                                    BxjĴ  
Z          @�����Ϳ�p�@�G�Bu�Ch������XQ�@3�
B��Cx=q                                    Bxj�Z  �          @�(���p����
@���Bpz�CfE��p��Y��@1�BQ�Cv�\                                    Bxj�   
�          @�(���  ���
@��BlffCa}q��  �^{@:=qBQ�CsE                                    Bxj�  �          @�  ��p��!G�@b�\BI�
Co����p��vff@G�A��HCx�                                    Bxj�L  T          @��R?���Q콸Q쿚�HC��=?��p���   ��Q�C�J=                                    Bxj�  T          @{��aG��qG�?&ffAQ�C�Y��aG��j=q��\)���
C�L�                                    Bxj�  �          @k���{�`  ?��
A��\C�=q��{�e�(���%�C�O\                                    Bxj+>  �          @5>��1�>�A��C�L�>��,(��W
=����C�XR                                    Bxj9�  "          @'
=�u�\)?.{AuG�C�E�u�!녿   �3�
C�P�                                    BxjH�  �          @{>aG��
=?�\AXz�C��>aG��
=���H�O�C�                                      BxjW0  �          @Dz���2�\?��A���C�O\���>{������C��
                                    Bxje�  �          @G��(���:=q?Y��A}�C��q�(���>�R�
=q�!p�C���                                    Bxjt|  "          @Z�H��
=�>{?���A�=qCxff��
=�I����{��G�Cy�                                    Bxj�"  �          @Vff�����8��?��\A��RCy8R�����B�\��p���=qCz0�                                    Bxj��  �          @Q녿0���G�?(��A:�\C��0���E�L���c
=C���                                    Bxj�n  �          @Y���O\)�B�\?���A���C
�O\)�L�;Ǯ�أ�C��                                    Bxj�  �          @a녿@  ���@��B1  C|��@  �O\)?���A��C�~�                                    Bxj��  T          @Y���L����H@{B$(�C{�{�L���J=q?h��A{
=C�)                                    Bxj�`  �          @HQ�J=q��Q�@(�B8�Cw�J=q�-p�?���A�  C}��                                    Bxj�  T          @Dz��G���@Q�B4(�C�����G��5?xQ�A�Q�C��                                    Bxj�  �          @8Q쾀  �  ?�  B��C�� ��  �2�\?z�A9p�C�z�                                    Bxj�R  
Z          @:=q��=q��H?��Bp�C��H��=q�6ff>��
@ə�C�>�                                    Bxj�  "          @G
=��Q��,��?�z�A�
=C�+���Q��C33=��
?�C���                                    Bxj�  T          @O\)?���2�\�z��%�C�n?���  ��  ���C�.                                    Bxj$D  
�          @W
=?�{��(��33�G�C��)?�{�#�
�&ff�S�RC�|)                                    Bxj2�  
�          @ ��?��׿�Q��(��S
=C��?��׿�{���\���C��H                                    BxjA�  	�          ?�{>�G����H?(�A��HC�h�>�G���=q��Q��I��C��                                    BxjP6  �          ?�z�?   �aG��aG��B�\C��)?   �333�z��C��f                                    Bxj^�  
�          @dz��zΌ��@��B0��CW{�z��Q�?���A�{Ce                                      Bxjm�  "          @�  �����@P��B\33Ccc׿���;�@��BffCr@                                     Bxj|(  �          @��׿�ff��@`  Brz�Cm�q��ff�A�@�HBp�Cz�{                                    Bxj��  "          @u���(���@<(�BJ�\Co����(��J=q?޸RA�  Cx�
                                    Bxj�t  
�          @l�Ϳ���\)@.{B=��Cd5ÿ��7
=?��Aҏ\CoǮ                                    Bxj�  T          @n{�s33��@/\)B?��Cwh��s33�O\)?�p�A�33C}�3                                    Bxj��  T          @mp��W
=�333@G�BG�C}\)�W
=�`��?Tz�AO\)C�J=                                    Bxj�f  
�          @i���s33�5@�
B	��C{}q�s33�]p�?!G�A�\C~��                                    Bxj�  
�          @o\)�\)�?\)@{B\)C��3�\)�i��?5A/�
C��\                                    Bxj�  
^          @W�>\�+�?�p�Bz�C�H>\�J=q>���@�C�ff                                    Bxj�X  �          @U�=#�
��ff?��BffC���=#�
��>�  @��C��R                                    Bxj��  T          @xQ�����33@'
=B*p�Cbuÿ����>�R?�(�A�\)Cl�q                                    Bxj�  �          @vff��z��G�@*=qB/�Cb�׿�z��>{?��A�(�CmE                                    BxjJ  X          @j�H��33��@G�B�Cj����33�E�?�ffA�Q�Cq�                                     Bxj+�  P          @x�ÿ��@'�B)ffCjuÿ��N�R?�{A�Q�Cr��                                    Bxj:�  
�          @tz��z��
=@(��B1�RCg�Ϳ�z��B�\?�p�A��HCqO\                                    BxjI<  
�          @j�H��ff�33@p�B({Cd�Ϳ�ff�9��?��A��
CnE                                    BxjW�  �          @6ff?&ff��33�L�����RC�Ф?&ff�333���H�NC��=                                    Bxjf�  �          @b�\?��?���!G��533B$Q�?��@#�
��=q���BV�                                    Bxju.  �          @u?��R?У��6ff�?��Bz�?��R@,(������\)BV�                                    Bxj��  �          @~�R@��?�33�/\)�.B&33@��@8Q���ǅBUp�                                    Bxj�z  T          @��
@#33?�{�;��5p�A�G�@#33@�R�z����B0                                    Bxj�   �          @��
@%?�{�9���2��A�Q�@%@p��33��\B.\)                                    Bxj��  �          @��@-p�?�Q��9���1p�A�@-p�@�
����Q�B!p�                                    Bxj�l  �          @���@33?��H�K��@�BG�@33@8���
=q����BN                                      Bxj�  �          @�=q@ff?���`���Zp�B�@ff@,���'����BPz�                                    Bxj۸  �          @���@?�
=�k��c�\A�{@@'��5�!{BM��                                    Bxj�^  �          @�p�@��?��
�Z=q�K
=B  @��@C33�� B[�                                    Bxj�  
�          @�(�@�?�(��X���K�HB �H@�@0���p��	\)BG=q                                    Bxj�  T          @��@�R?�G��Tz��C\)A���@�R@1G��Q����B@�                                    BxjP  
�          @�z�@ff?����Q��C��B+@ff@H���
=q��z�BaG�                                    Bxj$�  "          @�33?��@���L(��=33BI{?��@XQ�������
Bs�H                                    Bxj3�  T          @��@��@�
�G
=�6  B)��@��@J�H��(���z�BZ=q                                    BxjBB  T          @��
@
=q@�
�@  �-��B;z�@
=q@Vff��\��33Bd��                                    BxjP�  �          @�z�@�@��C33�1��B"(�@�@G
=��
=��z�BR�                                    Bxj_�  T          @�z�@33@�R�=p��*��B0\)@33@P�׿�\����B[                                      Bxjn4  �          @�ff@(Q�@G��=p��'
=B�
@(Q�@Dz��{�ř�BEQ�                                    Bxj|�  
�          @���@>{?����9���$�
AЏ\@>{@!G����܏\B!Q�                                    Bxj��  �          @�  @C33?���AG��)=qA��@C33@(��p���  B��                                    Bxj�&  �          @��@I��?G��E�.p�A^�\@I��?��R�\)�=qB �                                    Bxj��  �          @���@>�R?��,���
=A�ff@>�R@=q��\)��Q�B�H                                    Bxj�r  �          @�(�@I��?���&ff�33AУ�@I��@{��p���\)B��                                    Bxj�  "          @�33@C�
@�������B	p�@C�
@3�
��33�v�HB*G�                                    BxjԾ  "          @���@L(�@녿��R����Bz�@L(�@9���aG��7�
B){                                    Bxj�d  T          @��@\(�@z�ٙ���A�=q@\(�@%��333���B�                                    Bxj�
  T          @��@tz�?�Q쿬����A��H@tz�@ff�����z�A�                                      Bxj	 �  "          @�
=@��?�(������]�A�Q�@��?�ff����z�A���                                    Bxj	V  �          @�=q@n{?u�
�H��=qAhz�@n{?�\�˅��\)A�G�                                    Bxj	�  T          @�z�@aG�?:�H�;��A;�@aG�?�{�Q���
=A��H                                    Bxj	,�  "          @��H@n�R>�{�(Q��Q�@���@n�R?����G���A��                                    Bxj	;H  �          @��@��R�Ǯ���H���RC�T{@��R>k���  ��ff@J�H                                    Bxj	I�  "          @�ff@�(������G�����C�޸@�(�=u��\)��p�?^�R                                    Bxj	X�  
�          @�z�@vff��G���\)��  C��{@vff�   ���R�ظRC�Q�                                    Bxj	g:  
�          @��@�����=q����Z�HC���@����L�Ϳ��R���\C�g�                                    Bxj	u�  
�          @�p�@�녿#�
�!G��Q�C�� @�녾��ÿW
=�/�C�Ǯ                                    Bxj	��  T          @�Q�@��;�
=�L���$��C�H�@��ͽu�fff�9�C��q                                    Bxj	�,  
�          @�  @��
��R�Tz��+\)C��
@��
�k���G��P��C�z�                                    Bxj	��  
�          @�\)@�=q�B�\�n{�?
=C��@�=q���
��33�n�RC�ٚ                                    Bxj	�x  �          @���@��R�G���  �QG�C��@��R���R��(���33C�޸                                    Bxj	�  "          @�@�Q�W
=�^�R�6{C�e@�Q��
=����n=qC�&f                                    Bxj	��  T          @��@�  �k��8Q��C��q@�  �\)���
�W\)C�<)                                    Bxj	�j  �          @�@�
=��z�8Q��C�S3@�
=�E���\)�j�RC�˅                                    Bxj	�  "          @�=q@xQ��{��z��w�C�/\@xQ�˅��G��X  C��
                                    Bxj	��  �          @�  @k��ff�W
=�5�C�'�@k������
�_�C���                                    Bxj
\  "          @�ff@g��
=��33����C��f@g������z���
C��\                                    Bxj
  �          @�\)@l(���\��33���RC���@l(��޸R��\)�v�HC�e                                    Bxj
%�  
�          @�Q�@q녿�Q�W
=�4z�C�h�@q녿ٙ��u�PQ�C���                                    Bxj
4N  T          @��H@s33���#�
�ffC���@s33���ͿxQ��M�C��                                    Bxj
B�  
�          @���@X���%��\)�s�
C�P�@X���G����
��\)C�)                                    Bxj
Q�  �          @�=q@E��@�׾�z��z�HC���@E��)��������G�C���                                    Bxj
`@  T          @��@AG��H�ü���ffC��{@AG��7���G�����C�7
                                    Bxj
n�  "          @�33@H���?\)>���@���C�0�@H���8Q�\(��7\)C��
                                    Bxj
}�  "          @�33@Dz��A�?z�@�(�C��@Dz��@�׿&ff�	�C��                                     Bxj
�2  �          @�(�@J�H�>�R>�G�@�\)C�aH@J�H�:=q�B�\� Q�C��3                                    Bxj
��  T          @�=q@+��W�>�Q�@�33C�<)@+��P  �z�H�P��C�                                    Bxj
�~  
�          @��@(���X��>�=q@b�\C���@(���N�R�����g33C��q                                    Bxj
�$  �          @��@%�Vff>k�@J�HC��@%�J�H��=q�l��C��q                                    Bxj
��  �          @��@��]p�>��@c33C���@��R�\�����pz�C�8R                                    Bxj
�p  �          @�  @�H�^�R>8Q�@(�C�n@�H�Q녿�
=��
=C�<)                                    Bxj
�  �          @���@.�R�Tz�=�G�?�p�C��{@.�R�Fff��
=�~�RC��f                                    Bxj
�  
�          @�G�@.�R�S�
=��
?��
C���@.�R�E���H���\C���                                    Bxjb  �          @�  @#�
�X��=���?�=qC��f@#�
�J�H���H���RC�xR                                    Bxj  T          @�{@
=�[�>Ǯ@�33C�C�@
=�Tz�u�S�C��
                                    Bxj�  �          @�{@	���dz�>�@љ�C���@	���^�R�n{�L(�C�޸                                    Bxj-T  "          @�ff?��H�hQ�?G�A+�C�0�?��H�i���+��ffC�q                                    Bxj;�  �          @�
=@(��c33>�@�=qC��\@(��^{�h���H��C�                                      BxjJ�  T          @��\@Q��\(�?�Q�A��C���@Q��n�R<��
>��RC�޸                                    BxjYF  �          @��?�\�e�?��A��C�'�?�\����>\@�p�C��q                                    Bxjg�  �          @�z�?��R�Y��@33A�Q�C���?��R����?W
=A0z�C�(�                                    Bxjv�  �          @��
?����P��@
�HA�  C�o\?����u?E�A"�\C�t{                                    Bxj�8  �          @�z�?�Q��S33@�B �RC���?�Q��{�?h��A@  C���                                    Bxj��  T          @���?�G��I��?�A�\)C��R?�G��fff>��H@�C�                                    Bxj��  T          @�G�@��L(�?�{A�C�&f@��W��8Q��$z�C�l�                                    Bxj�*  �          @��?��@��@��B��C��H?��mp�?��A�  C��H                                    Bxj��  T          @�ff��Q��{@VffB[�
ChJ=��Q��>�R@=qB
=CtB�                                    Bxj�v  �          @�z�����\@Q�BA�CY0����7�@�B  Cf�                                    Bxj�  �          @�{�\)��Q�@O\)B<�CV#��\)�1�@�B��Cd#�                                    Bxj��  �          @��
���\@C�
B2�\C]  ��A�@�A�  ChW
                                    Bxj�h  �          @���!G���@333B&33CY=q�!G��4z�?��A�  Cd@                                     Bxj	  S          @����ff��@-p�B#G�C]��ff�<(�?޸RAƏ\Cg^�                                    Bxj�  �          @|���ff��\@�HB�RCc���ff�A�?�33A�  CkL�                                    Bxj&Z  T          @�G�����
=@��B�Cb�����E�?���A�33Cjk�                                    Bxj5   "          @����
=�&ff@)��BQ�Cg  �
=�Y��?�  A�  Cn0�                                    BxjC�  �          @�  ���#�
@<(�B/��Cl޸���^{?��A�33CtE                                    BxjRL  T          @�(���(���
@N�RBOCjW
��(��Fff@\)B��Ct�{                                    Bxj`�  �          @w
=��{�&ff@$z�B)(�Cv�)��{�W�?�
=A�{C{��                                    Bxjo�  �          @~�R�\�<(�@	��B(�Cr�H�\�aG�?k�AW�Cv�f                                    Bxj~>  �          @��Ϳ����b�\?���A�p�Cu�������u�=L��?#�
CwJ=                                    Bxj��  �          @����ff�b�\?�
=A���Cs  ��ff�n�R�L���.{Ct.                                    Bxj��  T          @�=q�.{�X��?�A���C�N�.{�s�
>�G�@���C��                                    Bxj�0  "          @}p��z��j�H?��
A�=qC����z��xQ������C��                                    Bxj��  
�          @xQ��(��c33?���A�
=C����(��u=u?fffC��\                                    Bxj�|  
�          @w
=<��
�c33?�{A��C�/\<��
�s33���
��{C�+�                                    Bxj�"  �          @w
=>����e�?�  A�z�C�&f>����r�\�\)�
=C���                                    Bxj��  T          @tz�?#�
�g�?��\Aw�C��)?#�
�o\)��Q���
=C��3                                    Bxj�n  �          @l(�=����N�R?�AQ�C��R=����Mp��.{�>�RC�ٚ                                    Bxj  �          @S�
>�\)��@)��BW=qC��>�\)�/\)?��
B(�C��                                    Bxj�  �          @K��L���%?��
B	��C��׾L���C�
?:�HAV{C��                                    Bxj`  �          @Fff�+��Q�?�G�B  C~W
�+��6ff?J=qAq�C�k�                                    Bxj.  �          @N{�z�H�
=@�B,�CuͿz�H�0  ?��
A��Czc�                                    Bxj<�  
Z          @W���R�\)@
=B5�C~�
��R�<(�?�33A�ffC��                                    BxjKR  �          @R�\�aG���@  B-z�C�O\�aG��@��?�  A��C��                                    BxjY�  �          @S�
�B�\�{@
=qB$G�C�Ф�B�\�E�?��A��C�>�                                    Bxjh�  �          @S33���H�C33?���A�33C�y����H�N�R���
��p�C���                                    BxjwD  �          @P  �p���:�H?}p�A��C|33�p���E���G�� ��C}{                                    Bxj��  T          @J�H��
=�(Q�?���A�\)Cu޸��
=�5=�\)?��CwxR                                    Bxj��  "          @Tz����\)@   BD�C[ff���
�H?���B33ChaH                                    Bxj�6  T          @7�>���3�
=���@G�C��>���*�H�fff��33C�f                                    Bxj��  �          @9��?z�H�*=q��G����C�'�?z�H��������Q�C�o\                                    Bxj��  �          @3�
?��R��33��G����C��?��R��=q�����+33C�'�                                    Bxj�(  �          @   ?��Ϳz�H��\)�#(�C�H�?��;�33��{�Bz�C��                                    Bxj��  �          @��?��Ϳu�Ǯ�z�C�~�?��;�33��ff�>��C��                                    Bxj�t  T          @7�?��ÿ�z��z��  C�
=?��ÿ=p��33�8�
C�ٚ                                    Bxj�  �          @Fff?�{��(���
=�
=C��?�{���
�
�H�6C�~�                                    Bxj	�  
�          @G
=@ �׿�z῝p���G�C�@ @ �׿�{�����Q�C��
                                    Bxjf  �          @Tz�?�z��33��
=�33C�?�z῾�R�'
=�Op�C��                                    Bxj'  �          @O\)?�{� ����#33C�  ?�{��z��*=q�`�RC�                                    Bxj5�  T          @O\)?fff�����'
=�[Q�C��?fff����@  k�C�,�                                    BxjDX  
Z          @H��?�
=��{�
�H�2=qC�8R?�
=�}p��+��p33C���                                    BxjR�  �          @HQ�@�Ϳ�\)�����C�u�@�;����
=q�0  C��                                    Bxja�  
�          @O\)@333�u��(�����C���@333����p����C�+�                                    BxjpJ  �          @Dz�@�R��=q�˅��  C���@�R�333�����ffC�H�                                    Bxj~�  �          @*�H?�ff���R�����ə�C�#�?�ff���\����
=C�(�                                    Bxj��  "          @.�R?�(��ٙ��fff���C��q?�(������
=���C���                                    Bxj�<  �          @9��?˅�(��xQ���ffC���?˅��p�����\C�H�                                    Bxj��  �          @5�?��H��(���=q��(�C���?��H��p���Q���
C��q                                    Bxj��  �          @)��?�Q��Q쿊=q��\)C�}q?�Q쿜(�������
C��                                    Bxj�.  T          @%?���У׿�R�d��C�޸?�����ÿ�����C�˅                                    Bxj��  
�          @G�?Q녿�(��
=q��Q�C��)?Q녿xQ�n{���C��                                    Bxj�z  T          @������z���W�Cp�3�����\)������Cl�                                    Bxj�   "          @ �׾�=q��R�L�Ϳ�C��3��=q��L����=qC�U�                                    Bxj�  
�          @8Q�?�(����R��ff�C�Q�?�(���(��������C�b�                                    Bxjl  T          @4z�?ٙ���\�\(���
=C�޸?ٙ��У׿��R��C��                                    Bxj   
(          @Mp�@��(����
��33C���@��
=����	33C���                                    Bxj.�  T          @J=q@��G��#�
�7
=C�� @����ͿTz��u��C���                                    Bxj=^  
�          @HQ�@Q��=q���+
=C�@ @Q���
��33��=qC���                                    BxjL  
�          @S�
@+���녿��
���RC�H�@+�������G���=qC��\                                    BxjZ�  T          @I��@ff�p��z��,  C���@ff��녿��
��33C��                                    BxjiP  �          @A�@��ff�.{�S
=C��@녿�  �����
=C��R                                    Bxjw�  "          @L��@"�\�У׿^�R��\)C��q@"�\��  ��\)��z�C���                                    Bxj��  �          @J=q@<�;��xQ���33C�` @<�;�������\)C���                                    Bxj�B  �          @P��@?\)�aG���Q����\C���@?\)>B�\��Q���\)@c�
                                    Bxj��  
�          @J�H?��;�?�G�A�C�*=?��Fff���
��\)C���                                    Bxj��  �          @J=q?��>�R?^�RA�  C��R?��E�#�
�>{C�Ǯ                                    Bxj�4  T          @A녾�\)�:�H>��HA�HC�H���\)�:�H�   �
=C�H�                                    Bxj��  
�          @O\)�����@��>#�
@7�Cyff�����9���W
=�p��Cx��                                    Bxjހ  "          @P  ��ff�0��?�G�A�33Ct�)��ff�<(�=L��?uCv:�                                    Bxj�&  
�          @L�;�\)�<(�>.{@S�
C�S3��\)�5��L���{
=C�9�                                    Bxj��  
�          @G�?n{�2�\�s33��G�C�1�?n{�z���Q�C�޸                                    Bxj
r  �          @J�H>��H�4zῧ���33C��{>��H�\)�
=q�-G�C�4{                                    Bxj  �          @J�H?aG��,(��������C��?aG��p���\)��{C��f                                    Bxj'�  �          @J=q?8Q��	��@	��B-��C�H�?8Q��.�R?�=qA��C�h�                                    Bxj6d  "          @XQ�?����R?�p�B�C�ff?���?\)?���A�  C��)                                    BxjE
  �          @c33?�Q��p�@�RB33C��?�Q��C33?���A�C���                                    BxjS�  �          @fff?���!G�@z�Bz�C�  ?���C33?��A��C���                                    BxjbV  �          @c�
?�
=�#33@
�HB�C�o\?�
=�G�?�p�A�Q�C�ff                                    Bxjp�  Y          @g�?�\)�ff@   B/G�C���?�\)�A�?�{A�(�C�'�                                    Bxj�  
�          @tz�?xQ��+�@!G�B&�C��\?xQ��U?��
A�C��                                    Bxj�H  T          @w
=?\�J=q?�
=A��
C��3?\�\��>�33@�  C��)                                    Bxj��  "          @vff?�\)�Mp�?z�HAlQ�C��?�\)�W
=���Ϳ��HC���                                    Bxj��  �          @u?�33�]p�<��
>�33C���?�33�R�\��=q��
=C�N                                    Bxj�:  T          @z�H?�G��_\)�����RC�aH?�G��P�׿�  ���C�.                                    Bxj��  
�          @|(�?�\)�\(�������{C�@ ?�\)�J�H��������C�Ff                                    Bxj׆  �          @xQ�?�33�U=u?Tz�C��=?�33�L(���G��r�HC�aH                                    Bxj�,  
�          @vff?���U>�?�\)C��)?���Mp��p���b=qC�:�                                    Bxj��  
�          @y��@Q��Dz�}p��n=qC�U�@Q��&ff��z���33C���                                    Bxjx  �          @�  @�
�Dzῇ��vffC�q�@�
�%���(���\)C��H                                    Bxj  "          @�Q�@��I���333�!G�C�w
@��1G���z���p�C�B�                                    Bxj �  �          @~�R@�I���!G��C�H�@�333�˅��(�C��{                                    Bxj/j  
�          @\)@�
�H�ÿL���9��C�7
@�
�.�R�޸R�ϮC�'�                                    Bxj>  �          @\)@(��C33�O\)�:�HC�S3@(��(�ÿ�(���ffC�W
                                    BxjL�  �          @y��@���<�ͿY���J�RC�� @���"�\��p���\)C���                                    Bxj[\  T          @z�H@{�AG��������C�1�@{�!G�� ����
=C���                                    Bxjj  �          @w
=@G��N�R����z�C���@G��9�����
��p�C�ff                                    Bxjx�  �          @mp�?�Q����p��
=C���?�Q�z�H����?�RC��                                    Bxj�N  �          @s�
?�\)���\�C�
�V�C��H?�\)����S�
�q  C��q                                    Bxj��  "          @l(�?�����AG��\(�C�k�?���G��Mp��rQ�C�O\                                    Bxj��  �          @j=q?�p������1��E��C�+�?�p������C33�b  C�Q�                                    Bxj�@  �          @u@�
���R�)���/��C���@�
�(��>�R�M(�C���                                    Bxj��  �          @tz�?�Q����1G��A��C��3?�Q�Y���L(��m33C��                                    BxjЌ  T          @o\)?(�ÿ�
=�^�R��C��q?(�ýL���j�H��C��=                                    Bxj�2  T          @p  ?녿�{�\(��C�y�?녾u�l(� �HC���                                    Bxj��  
�          @g
=>\����W
=��C�'�>\�u�c33§C���                                    Bxj�~  "          @j�H?zῡG��X��k�C�q�?z�.{�g
= �HC��
                                    Bxj$  �          @h��?p�׿�
=�L���wz�C���?p�׾Ǯ�^�R�C���                                    Bxj�  T          @g
=?���\�.�R�E
=C��q?���\)�N{�}p�C��                                    Bxj(p  T          @e?�=q�(�����'ffC��
?�=q�����@���effC��                                    Bxj7  "          @g�?z�H�"�\���$(�C���?z�H�ٙ��A��c�HC���                                    BxjE�  T          @e�?����.{��Q��z�C���?���� ���*�H�C�C�]q                                    BxjTb  "          @g
=?n{��p��G��r�
C��?n{����[�  C���                                    Bxjc  T          @(Q�>�G�<#�
�#33 �=?�>�G�?G���H��Br��                                    Bxjq�  �          @&ff?���u�   ��C��?��?����(�B"\)                                    Bxj�T  
�          @p�?녿��33(�C�p�?�>\)�
=\A[�                                    Bxj��  
�          @��>�Q����\��C�5�>�Q�>8Q��ff k�A�(�                                    Bxj��  �          @>��
��Ϳ��Rp�C�
=>��
=�\)��
¢A:=q                                    Bxj�F  �          @�\?
=�\)�Q���C�>�?
==�Q����\)A�\                                    Bxj��  T          @^�R?=p��n{�P���C��?=p�=��
�X��aH@��
                                    Bxjɒ  �          @h��?녾�(��dz�ffC��
?�?&ff�b�\�BB�                                    Bxj�8  �          @N�R    >��
�L(�¨�B�
=    ?�  �<��L�B�                                      Bxj��  "          @�=q���?:�H�~{#�B��
���?�\)�e��wffB��)                                    Bxj��  "          @�ff�\)�#�
��{°33C6���\)?�p���Q��RB��                                    Bxj*  �          @s�
>����R�q�©��C�9�>�?Q��l����B�\)                                    Bxj�  
�          @\(�>#�
��G��Z=q¤p�C���>#�
?z��X����B�z�                                    Bxj!v  
�          @�z�@  �=p��|(�#�C`�ÿ@  >�ff�~�R\)C)                                    Bxj0  �          @~{�\)�L���u�(�C����\)>�Q��y��¨�B�p�                                    Bxj>�  "          @aG�    �.{�Z�H�\C�'�    >�33�^{¨�B���                                    BxjMh  
�          @L��?��\��\)� ���9�HC�?��\�u����n��C�B�                                    Bxj\  
�          @G�@���Ϳ�R�7�
C���@���
=��p���p�C��\                                    Bxjj�  "          @P  @���p��L���h��C�˅@����
�Tz��t(�C��3                                    BxjyZ  !          @Tz�@.{�޸R�!G��1C���@.{���R��{���C��                                    Bxj�   �          @S�
@!녿���Q�����C�E@!녿��\����33C��                                    Bxj��  "          @W�@�G��Q��`��C��@��Q쿸Q���z�C�,�                                    Bxj�L  �          @Vff@z���H�������C���@z´p�����	��C���                                    Bxj��  "          @[�@�H���u��G�C�Z�@�H��G����
���C���                                    Bxj  �          @XQ�@�ÿE���G��Q�C��@�þk���z����C�AH                                    Bxj�>  �          @U�@+���\)������RC��@+��!G������
=C�k�                                    Bxj��  �          @XQ�@*�H���Ϳ�G�����C���@*�H�\(������C��                                    Bxj�  �          @\��@2�\��녿������HC�Ǯ@2�\�p�׿�(���  C��
                                    Bxj�0  �          @\��@$z����������z�C�e@$z���ÿ�=q��p�C�P�                                    Bxj�  �          @\��@.{�����У���\C�  @.{�.{��z���HC���                                    Bxj|  T          @_\)@#33�޸R����ә�C���@#33���R��(���C��                                    Bxj)"  
�          @`��@p������\���\C��3@p���
=������33C���                                    Bxj7�  �          @aG�@�
�   �G��MC�Y�@�
��������z�C�P�                                    BxjFn  �          @]p�@
=�)���u���C�E@
=�!녿O\)�\Q�C���                                    BxjU  �          @c33@*=q��\�����C��f@*=q���������C��)                                    Bxjc�  �          @hQ�@��,��>��@33C�H@��(Q��R�z�C�^�                                    Bxjr`  
�          @g
=@Q��,(�>B�\@Dz�C���@Q��(�ÿ����C�                                    Bxj�  Z          @o\)?���@��?�ffA�p�C�
?���K�>.{@#33C�aH                                    Bxj��  �          @u@-p���  ����Q�C���@-p�������C���                                    Bxj�R  �          @|(�@7
=����������C�S3@7
=��z������C��3                                    Bxj��  "          @~�R@:=q�(���{���RC���@:=q��\��  ��C�|)                                    Bxj��  �          @~�R@7���Ϳ�
=��z�C��@7��G�����C�n                                    Bxj�D  "          @{�@7
=�%�����߮C���@7
=�ff����(�C�P�                                    Bxj��  T          @��H@<(��  ���H��=qC�P�@<(���Q������HC��                                    Bxj�  �          @��
@*�H�:=q�s33�X  C�C�@*�H�"�\��(����
C�<)                                    Bxj�6  �          @��@333�:=q�c�
�F�RC��q@333�#33��z�����C��q                                    Bxj�  "          @��R@G��\)��  ���HC��)@G��33�����(�C�`                                     Bxj�  �          @�Q�@Fff�{���R���\C��)@Fff��(��
=���HC��\                                    Bxj"(  "          @��R@Mp��#33�k��IC���@Mp���Ϳ�=q���RC���                                    Bxj0�  �          @��
@Mp��!G��Ǯ���
C���@Mp��   �(���33C�
=                                    Bxj?t  �          @���@U�$zῘQ��{�C�8R@U�	�������C��H                                    BxjN  T          @���@L(��:=q�G��"�HC���@L(��%������=qC�~�                                    Bxj\�  �          @���@>{�>�R�
=� Q�C�s3@>{�-p�������C��{                                    Bxjkf  �          @��@&ff�N{�c�
�A�C�o\@&ff�7
=��p�����C��                                    Bxjz  T          @�
=@!��HQ쿝p����C�~�@!��+�����z�C���                                    Bxj��  T          @��R@   �e�G��,z�C��f@   �P  ��p����HC��=                                    Bxj�X  �          @�p�?�
=�`  ������z�C��=?�
=�@  �G��  C���                                    Bxj��  
�          @�  @
=q�b�\��ff��(�C��@
=q�R�\��������C��H                                    Bxj��  "          @�(�@�\�S�
?E�A,��C�T{@�\�Y������Q�C���                                    Bxj�J  "          @��?���W�>��@\)C���?���R�\�B�\�5p�C���                                    Bxj��  T          @��
?���w��\)��p�C�W
?���l(������C�                                    Bxj��  �          @�{?��~{=u?J=qC�:�?��u��G��ap�C�}q                                    Bxj�<  
�          @��@�R�`  ���
��=qC�9�@�R�Vff��G��ap�C��=                                    Bxj��  �          @�(�?�
=�s�
���陚C�L�?�
=�a녿\��C��                                    Bxj�  "          @�?�ff�w
=�n{�M�C�E?�ff�^�R�����C�8R                                    Bxj.  "          @��?aG��p  ���H��  C���?aG��J=q�)���C�Ǯ                                    Bxj)�  
Z          @�녽�G���G�>8Q�@#�
C�@ ��G��|(��fff�L��C�<)                                    Bxj8z  
�          @���>\)��Q�?�z�A�p�C��>\)���>���@��\C���                                    BxjG   �          @��H?@  �qG�?�p�AمC��
?@  ���?h��AB{C�{                                    BxjU�  
�          @�\)>��w�?���A���C�c�>����?�@�p�C�(�                                    Bxjdl  
�          @��>k��b�\@Q�A�C��)>k��~{?�\)A{�C���                                    Bxjs            @�  �W
=�^{@�\B�C�'��W
=�|��?�ffA��HC��
                                    Bxj��  T          @���8Q��Z�H@p�BC�~��8Q��|��?�p�A��C���                                    Bxj�^  
Z          @��R    �`  @z�B  C���    �\)?���A�z�C��)                                    Bxj�  
�          @��R?�\�Z=q?�z�A癚C�G�?�\�r�\?s33A_
=C���                                    Bxj��  �          @�ff?����l(�?��A|��C��q?����w
=>�?���C�Ff                                    Bxj�P  
�          @���?��g�?Q�A8z�C�E?��mp������\C��q                                    Bxj��            @��@G��fff>��@��C��@G��e��
=q��C��3                                    Bxjٜ  �          @�(�?��c�
?��A�(�C�!H?��qG�>��
@�p�C��                                    Bxj�B  �          @�33?+��a�?��Aܣ�C�\)?+��y��?fffALz�C��q                                    Bxj��  �          @�{?��g
=?˅A�p�C��\?��y��?��AG�C�P�                                    Bxj�  T          @�{?.{�_\)@A�\)C�z�?.{�z�H?�\)Az�HC��f                                    Bxj4  "          @�
=?B�\�vff?���A��C���?B�\���\>�{@�{C�C�                                    Bxj"�  "          @�=q?Q��w
=?��HA��
C�  ?Q���p�?(��A��C��\                                    Bxj1�  T          @�{?����s�
?Y��A@  C���?����y���#�
�\)C���                                    Bxj@&  
�          @��׽����\(�@�RB\)C�,ͽ����}p�?\A��C�H�                                    BxjN�  �          @�  �(��U@7�B!  C��\�(��}p�?�A̸RC��q                                    Bxj]r  T          @���ff�u�?�\)A��C~�)��ff��?Tz�A.=qCٚ                                    Bxjl  
�          @�����u�?�G�A�
=C}
=�����H>�@�p�C~�                                    Bxjz�  
�          @�  �u�|(�@��A�p�C�@ �u��33?��AZ�\C�j=                                    Bxj�d  
�          @��R���R����?�{A�
=C��쾞�R��z�?E�AffC��)                                    Bxj�
  �          @�33�
=q����@A�z�C�7
�
=q��ff?}p�AF�HC���                                    Bxj��  
�          @�ff�����h��@-p�B  C}@ �������R?ٙ�A��\Ch�                                    Bxj�V  �          @�{�:�H�vff@#�
B�C��R�:�H���
?�  A��C�:�                                    Bxj��  T          @�z���p��@*=qB33C�׿����?�\)A�  C��                                    BxjҢ  
�          @�z��G��z=q@�A�z�C����G���(�?�ffA��C�%                                    Bxj�H  �          @�z�O\)��ff?�A���C����O\)����?0��A	�C���                                    Bxj��  
�          @���333��33@��AمC�0��333��Q�?��
ALz�C��q                                    Bxj��  �          @�{��R����@'
=A��C�����R��p�?�p�A���C�7
                                    Bxj:  T          @�33��{����@*�HB�C��q��{���?���A�C��                                    Bxj�  
.          @��H��G��C�
@_\)BA��C��3��G��u@'
=B��C�                                    Bxj*�  
�          @��;�׿xQ�@��RB�B�Ct0���׿��R@u�Bwz�C�`                                     Bxj9,  
�          @��H��\�\@~�RB�{C{Q��\�{@^{BX�C�q                                    BxjG�  �          @�G��   �   @qG�B_��C�N�   �W
=@A�B&�C���                                    BxjVx  
�          @��;��
�#33@xQ�Ba��C�s3���
�\(�@G�B(\)C�\)                                    Bxje  T          @�\)�aG��3�
@XQ�BC�
C|��aG��c�
@$z�B=qC��                                    Bxjs�  T          @�p��E��ff@i��B_z�C{�
�E��K�@<��B(  C�1�                                    Bxj�j  �          @�{�.{�	��@r�\Bl��C|c׿.{�A�@H��B5ffC���                                    Bxj�  "          @�=q��p���Q�@�33B���C8\��p�����@|��B��3C]��                                    Bxj��  
�          @�녿�
=�!G�@�Q�B�ffCK���
=����@n�RBp=qCd33                                    Bxj�\  
�          @�녿��J=q@\)B��3CQ�����  @j�HBi�
Cf�3                                    Bxj�  
�          @�����
=�}p�@y��B�CV�=��
=��@b�\B_��Cis3                                    Bxj˨  
�          @�G����>aG�@��B���C*����J=q@\)B�\)CRxR                                    Bxj�N  "          @��׿��
?5@�  B��RC޸���
��z�@���B�\C@�)                                    Bxj��  
�          @�\)��p�?�ff@y��B���Cp���p�=u@�G�B�#�C18R                                    Bxj��  
�          @�
=��G���p�@mp�By\)C[��G��ff@R�\BO�CjG�                                    Bxj@  
�          @��ÿ���@G�@<(�B4�HC ������?��R@Y��B]�\C�                                    Bxj�  "          @�  ���>aG�@y��Bt�C.���ÿ=p�@uBn
=CG\                                    Bxj#�  T          @�����\��@�G�B�  CA:���\��G�@�G�Bg�\CX��                                    Bxj22  
�          @��R��(����@�
=B�ffCC��(�����@}p�Bf\)CZ�)                                    Bxj@�  "          @��������@s33BoffCW������@W�BH\)Cf��                                    BxjO~  �          @�z��ff���@U�BW=qCfaH��ff�(Q�@1G�B(�HCop�                                    Bxj^$  "          @���\��R@4z�B+Q�Cjp���\�E@��A���Cp.                                    Bxjl�  �          @�{��\)�G�@�A�p�Co\��\)�aG�?��HA�\)Cr                                    Bxj{p  "          @�=q��G��>�R?�A�Cs)��G��W
=?��A�=qCu�                                    Bxj�  T          @~{�z�H�(�@@  BC(�Cx&f�z�H�E@z�BQ�C|u�                                    Bxj��  
�          @�  �#�
�O\)@�\B�HC��q�#�
�l��?�Q�A��
C��=                                    Bxj�b  
�          @�(�>��H��  ?�(�A�(�C��H>��H���>�@�z�C�O\                                    Bxj�  (          @��\?������?�@ӅC�C�?����z������C�E                                    BxjĮ  
�          @�33?�����
?�@�33C�1�?����(��   ����C�.                                    Bxj�T  
�          @�=q?\����?
=@�C�Ǯ?\���þ�ff���RC���                                    Bxj��  "          @���?�33���H�\)��p�C�l�?�33����
=�r�\C���                                    Bxj�  T          @�{?J=q��(�=L��?�RC�b�?J=q���׿s33�PQ�C���                                    Bxj�F  
Z          @�{?�G���p�?s33A=C�l�?�G����׾\)��
=C�J=                                    Bxj�  
�          @�p�?����?@  A��C���?����\)��{��{C��                                     Bxj�  �          @���?�G����\(��)��C��\?�G���33��33��
=C���                                    Bxj+8  
�          @��R?E���G�?L��A�RC��=?E���33�����n{C���                                    Bxj9�  �          @���>����{�@��A�
=C�7
>������H?���AuC�H                                    BxjH�  T          @�\)>�������?�
=A��
C�P�>������
?h��A;�C�"�                                    BxjW*  
�          @�Q�?��R��\)<��
>��RC���?��R����z�H�J�RC��                                    Bxje�  T          @�?�p����׼#�
�#�
C�Ф?�p���(�����L  C�!H                                    Bxjtv  
�          @�z�?�33���>8Q�@
�HC�z�?�33����c�
�'�C���                                    Bxj�            @��?����33��z��XQ�C��?�����Ϳ�����z�C��\                                    Bxj��  "          @��\?��
���Ϳn{�)p�C�|)?��
����� �����RC�B�                                    Bxj�h  �          @��\?�
=���H����<��C�l�?�
=���R�ff�¸RC�T{                                    Bxj�  
�          @�?�
=��p��z�H�8z�C�˅?�
=����   ���C���                                    Bxj��  (          @�z�@33����
=��ffC�9�@33�~�R�������C��q                                    Bxj�Z            @�(�@����  �5���C�H@���~{��Q����\C�ٚ                                    Bxj�   
�          @���@���Q�Tz����C���@���{��\)���RC���                                    Bxj�            @�ff@33������  �1��C�n@33��{��\��G�C�j=                                    Bxj�L  �          @�ff@Q���(���ff�9C�]q@Q���  �
=��
=C�P�                                    Bxj�  
/          @�@
�H��녿����I�C��3@
�H��p��
�H��{C���                                    Bxj�            @�\)@"�\�����(��Xz�C��
@"�\��  ��R��G�C�4{                                    Bxj$>  "          @�(�@'
=������R����C��@'
=�l(�����{C���                                    Bxj2�  T          @�p�@-p����R��G��bffC�Z�@-p��s33�p����C���                                    BxjA�  �          @��
@+���G��!G���C��@+����׿�\)��  C��R                                    BxjP0  T          @��\@:=q�����R�߮C���@:=q�vff�Ǯ��\)C��\                                    Bxj^�  �          @��\@W��i���z�H�3
=C�Q�@W��Tz����p�C��                                    Bxjm|  "          @��@`���a녿Y����C�q�@`���N�R��33���HC��{                                    Bxj|"  T          @���@p���I����p��a�C�
=@p���1G���
=���
C���                                    Bxj��  
�          @��@s�
�@  ����EC��@s�
�*�H��p���\)C�~�                                    Bxj�n  �          @�ff@Y���c�
������{C��\@Y���W���(��eG�C���                                    Bxj�  
�          @��@8���|(������Q�C�  @8���n�R����xz�C���                                    Bxj��  �          @�@>�R�w
=��Q�����C���@>�R�j�H��  �m�C��\                                    Bxj�`  �          @��Ϳu���H?���A�Q�C�  �u���\>�@�(�C�h�                                    Bxj�  �          @�녿�=q����@	��A�ffCz�Ϳ�=q��p�?��
A5�C|(�                                    Bxj�  
Z          @�ff?��R�/\)�#�
�(��C��{?��R�*=q�+��7�C�`                                     Bxj�R  "          @��H@7
=����>�  @;�C�� @7
=�~{�.{��
=C���                                    Bxj��  T          @�������?���A�33C�` ���*=q?J=qA�{C��f                                    Bxj�  T          @�{��{?\)@��
B+=qC-.��{���@�(�B+��C9��                                    BxjD  
Z          @�
=���?�@�{B-��C-
=������@�ffB.(�C9�=                                    Bxj+�  T          @θR���>���@�
=B.��C/!H����#�
@�{B-\)C;�H                                    Bxj:�  
�          @�  ����>��
@�(�B)C0(����ÿ333@��HB'�C<O\                                    BxjI6  
�          @У����>#�
@��\B&C2�����W
=@�  B#
=C=�=                                    BxjW�  �          @Ϯ��(��8Q�@���B$�
C6!H��(���z�@��
B�\CAc�                                    Bxjf�  �          @����Q�8Q�@���B((�C6+���Q쿔z�@�z�B �RCA�q                                    Bxju(  "          @��
��z�<�@��B,C3����z�z�H@�  B'\)C?�H                                    Bxj��  �          @�����
=�u@��B)��C4Ǯ��
=��ff@�B#ffC@�{                                    Bxj�t  T          @�z����
�.{@���B$33C6)���
���@xQ�B
=CAG�                                    Bxj�  
�          @�=q��\)����@l(�B��C8+���\)��@aG�B�CB��                                    Bxj��  
�          @�\)��G���p�@R�\B=qC90���G���{@G�B�CCc�                                    Bxj�f  
�          @�Q���Q�u@AG�B  C?�R��Q��\)@.{A���CG��                                    Bxj�  �          @�(���33�W
=@HQ�B{C>c���33���@6ffA�CFxR                                    Bxj۲  T          @��R��
=�=p�@HQ�B
=C<�H��
=��
=@8Q�A�Q�CD��                                    Bxj�X  
�          @�z������:�H@P  B��C=���������@?\)B G�CE��                                    Bxj��  �          @����z�0��@S33B�C<s3��zῷ
=@C�
B CE{                                    Bxj�  "          @��\��z��ff@`  B�
C9����z῞�R@S�
B
\)CB��                                    BxjJ  �          @��\��  ���@j�HB\)C90���  ���R@^�RB�HCCY�                                    Bxj$�  
�          @�{�����\@r�\B�C8Ǯ������p�@fffBp�CC.                                    Bxj3�  T          @�
=��녾�33@u�B�C8Y���녿��H@i��B�
CB޸                                    BxjB<  "          @�\)���þ���@xQ�B"{C7���ÿ�@mp�B��CB��                                    BxjP�  T          @�=q�������@xQ�B  C7�������
=@mp�B�RCB0�                                    Bxj_�  �          @�
=�����W
=@�
=B+(�C6��������z�@��B#G�CBY�                                    Bxjn.  �          @θR���׽L��@��
B6ffC4�����׿���@�\)B/��CA�H                                    Bxj|�  �          @�p���Q콏\)@�=qB5z�C4�)��Q쿎{@�{B.�RCA                                    Bxj�z  T          @�����
=��\)@��\B6C4���
=��{@�{B/�HCA�3                                    Bxj�   "          @�z���ff����@��HB7\)C5@ ��ff���@�{B0(�CBQ�                                    Bxj��  T          @��
��
=<��
@���B5z�C3���
=��  @�p�B/��C@��                                    Bxj�l  �          @��
�����
@��\B7��C4E�����@��RB1z�CAs3                                    Bxj�  T          @θR��33���
@���B1��C7�3��33��=q@��\B(�\CD.                                    BxjԸ  
�          @�
=��33����@���B1�RC8����33��@�=qB'�\CE�                                    Bxj�^  �          @Ǯ��G���p�@���B,�RC8����G���=q@�=qB#
=CD^�                                    Bxj�  T          @�(����R��Q�@�B,=qC8�����R��ff@\)B"�CD:�                                    Bxj �  T          @�33��{�.{@��HB1�C6E��{���@|(�B)ffCB�\                                    BxjP  T          @�33���\�k�@{�B(��C7{���\��\)@qG�B �\CBs3                                    Bxj�  �          @�p����;k�@|��B'�RC6�q���Ϳ�\)@s33B��CBG�                                    Bxj,�  
�          @���(����R@~�RB(��C7�q��(�����@s�
B��CCG�                                    Bxj;B  �          @��
��33���R@{�B(�C8
=��33��Q�@p��BG�CCE                                    BxjI�  �          @��H��33���
@xQ�B&C85���33��Q�@mp�B��CCL�                                    BxjX�  
�          @�����녾�
=@s�
B%33C9�{��녿��\@g�B(�CDh�                                    Bxjg4  "          @�����{�\)@c33B =qC;�{��{����@UB�CE�\                                    Bxju�  �          @������H��(�@k�B'(�C:���H��  @^�RB�CE                                      Bxj��  T          @����녿�@a�B#{C;k���녿��@Tz�B�\CE�                                    Bxj�&  T          @������׾�G�@e�B&Q�C:33���׿�p�@X��B�CE{                                    Bxj��  
�          @��\�~�R��ff@`��B$�C:��~�R���R@Tz�B{CE=q                                    Bxj�r  "          @�\)��{��@`  B��C;�3��{���@Q�B
=CE�                                    Bxj�  
(          @��\���׿(�@a�B=qC<#����׿��@S33BQ�CF�                                    Bxj;  T          @�  ��{��@aG�B(�C;�q��{����@S33B�CEٚ                                    Bxj�d  
�          @������Ϳ�@i��B$\)C;B����Ϳ��@\(�B�CE޸                                    Bxj�
  T          @�����(��5@g
=B#{C=�\��(���G�@W
=B�CH{                                    Bxj��  �          @�����Q�L��@aG�B"C?:���Q����@P  B  CIY�                                    BxjV  "          @����|(��B�\@g
=B({C>޸�|(���ff@VffBp�CI�                                     Bxj�  T          @�(��xQ�8Q�@j=qB+z�C>���xQ���
@Y��B  CIz�                                    Bxj%�  �          @���z�H�8Q�@_\)B$�C>u��z�H���R@O\)B�\CH��                                    Bxj4H  T          @�  �~{�p��@R�\BffCAW
�~{��z�@?\)B33CJ��                                    BxjB�  "          @����|�;�@\��B#�HC:�3�|�Ϳ�p�@P��B��CEL�                                    BxjQ�  T          @����}p���Q�@k�B+�RC5T{�}p��k�@dz�B%�CA�                                    Bxj`:  �          @�G��\)�#�
@^{B$  C6L��\)�p��@UB�HCA8R                                    Bxjn�  �          @�����
�k�@_\)B!  C7.���
��G�@VffBG�CA�                                    Bxj}�  �          @�����\��\)@]p�B �
C7�����\��ff@S�
Bz�CBh�                                    Bxj�,  "          @��H��������@]p�B!�C8�
��������@S33B�CC#�                                    Bxj��  
�          @���y��>�{@fffB*p�C/��y����@eB)�C;                                      Bxj�x  M          @�ff�s33=�\)@`��B+{C3�s33�:�H@\(�B&�\C>ٚ                                    Bxj�  �          @�z��q녾aG�@^{B)�C7^��q녿�  @U�B!��CBǮ                                    Bxj��  �          @�z��p  �
=q@]p�B)�\C<8R�p  ���@P  B{CG5�                                    Bxj�j  "          @�
=�|(��#�
@[�B$\)C4��|(��G�@UB(�C?8R                                    Bxj�  
�          @�Q���ff>B�\@c�
B!
=C1p���ff��R@`��B\)C<aH                                    Bxj�  �          @��\����>B�\@c33B�C1��������R@`  B�C<+�                                    Bxj \  �          @���s33��{@_\)B)��C9��s33��{@U�B 33CD\)                                    Bxj   �          @��H�mp�����@^{B+�HC9��mp�����@S33B"  CD�
                                    Bxj �  	�          @�Q��|�;.{@^{B%=qC6��|�Ϳs33@UB�CA��                                    Bxj -N  "          @�
=��{�8Q�@`��B��C6k���{�u@XQ�BC@��                                    Bxj ;�  "          @�=q��녽���@a�B33C5\)��녿c�
@Z=qB33C?�R                                    Bxj J�  "          @������.{@fffB�C65�����z�H@^{B\)C@�)                                    Bxj Y@  �          @����녾W
=@j=qB!{C6����녿��
@aG�B�RCA^�                                    Bxj g�  �          @�����(���\)@p��B"\)C7�)��(���\)@fffB33CBL�                                    Bxj v�  �          @��R�xQ��G�@\(�B%��C:s3�xQ쿙��@P��B�HCE:�                                    Bxj �2  
�          @����\    @h��B 33C4����\�Q�@c33BG�C>                                    Bxj ��  "          @��\��  ���
@l��Bz�C4B���  �Y��@fffBp�C>��                                    Bxj �~  T          @�G���
=<#�
@k�B��C3���
=�Q�@eB
=C>s3                                    Bxj �$  �          @��H��p���R@q�B!�C;����p�����@b�\B33CF:�                                    Bxj ��  T          @�����
=��R@s33B z�C;����
=���H@c�
B��CF!H                                    Bxj �p  �          @�{��Q�\)@tz�B 33C;)��Q쿴z�@fffB
=CE\)                                    Bxj �  
�          @�
=��G��z�@uB {C;@ ��G���
=@g
=B��CE}q                                    Bxj �  T          @���p���@k�B!��CC����p���(�@Tz�B=qCMJ=                                    Bxj �b  �          @�ff����p��@k�B!{C@}q�����  @XQ�BQ�CJaH                                    Bxj!	  
�          @���G��@  @h��Bz�C=�3��G��Ǯ@W�B�HCG�                                    Bxj!�  T          @��������aG�@dz�B�C?��������z�@Q�B�RCI#�                                    Bxj!&T  �          @������n{@hQ�BC@k������p�@U�B{CJ8R                                    Bxj!4�  "          @��\��
=�5@c�
B{C=� ��
=���R@S�
B�CG}q                                    Bxj!C�  
�          @�  ��G�����@Tz�B{C8���G�����@J=qB�CB�)                                    Bxj!RF  	`          @��\���;���@UB  C8(����Ϳ�ff@K�B�\CB0�                                    Bxj!`�  
�          @�{�����Q�@>�RB(�C8�=������\@5�BCA��                                    Bxj!o�  �          @��
��
=��\)@8Q�B	Q�C4�q��
=�8Q�@2�\BffC=��                                    Bxj!~8  "          @�
=�s33�33@W�BCPff�s33�0��@3�
A�\)CX�                                    Bxj!��  
�          @�=q��G����H@Z�HB�CJ�R��G��(�@<��A��
CS+�                                    Bxj!��  T          @�  ���Ϳ�
=@S�
B{CG��������@9��A��RCOO\                                    Bxj!�*  �          @�
=�����Q�@6ffB�\C8������  @,��A��HC@�3                                    Bxj!��  T          @���G��=p�@333B�C=����G���{@#�
A��HCE��                                    Bxj!�v  T          @�z�����W
=@7�B �C>�{�����(�@&ffA��HCF)                                    Bxj!�  "          @�����׿}p�@K�B��CA)���׿�Q�@7�A�CI�
                                    Bxj!��  T          @�
=����{@:=qB �CE�����p�@!G�A��CL)                                    Bxj!�h  �          @�p����\��p�@\)Aڣ�CE����\� ��@A��CK�                                    Bxj"  
�          @��H��녿��R@�A�\)CF���녿�p�?�
=A��
CKp�                                    Bxj"�  
�          @�33���Ϳ�p�@)��A�=qCF�������33@  A���CL��                                    Bxj"Z  "          @�����{��33@"�\A���CE����{����@	��A��CK��                                    Bxj".   �          @�����\)���
@4z�B =qCG�3��\)���@��A�z�CN��                                    Bxj"<�  
�          @�
=�vff�Ǯ@I��B�RCJ��vff�\)@-p�A��RCR+�                                    Bxj"KL  M          @�=q�mp��У�@\(�B!Q�CK� �mp��Q�@>�RB=qCT��                                    Bxj"Y�  S          @�33�w�����@<��B��CJ)�w��p�@ ��A�33CQ��                                    Bxj"h�  
�          @��\�QG��У�@aG�B/�HCN}q�QG����@C�
B33CX.                                    Bxj"w>  
�          @���\�Ϳ޸R@U�B"�HCN�\�\���p�@5B\)CWxR                                    Bxj"��  "          @�(��^{��Q�@W
=B#�CN��^{��H@8Q�B��CV�
                                    Bxj"��  
�          @�33�[���G�@VffB#��CO��[���R@7
=B(�CW�\                                    Bxj"�0  �          @�Q��`  �ٙ�@FffB=qCM��`  ��@(Q�A�33CV{                                    Bxj"��  
�          @�\)�^�R��@7
=B��CR�)�^�R�+�@�
A�  CY��                                    Bxj"�|  �          @����G
=�'
=@   A���C[�q�G
=�G�?�=qA��HCa�                                    Bxj"�"  �          @�\)�e��G�@<��B��CN��e���@{A�ffCU�f                                    Bxj"��  
Z          @�=q�g
=�˅@J�HB�HCK���g
=�G�@.{B�
CT+�                                    Bxj"�n  "          @�  �c33���R@L(�B�CJ�c33��@0��B{CS�{                                    Bxj"�  
�          @���_\)��G�@X��B)CG���_\)� ��@@��B(�CQ�                                    Bxj#	�  
�          @���~{��Q�@W�B33CG���~{��@<��B{CP�                                     Bxj#`  �          @��H�s33��33@c�
B'��CD�H�s33��Q�@L��B�\CO{                                    Bxj#'  T          @�=q�~{���@h��B#��CF�H�~{���@N�RBffCPT{                                    Bxj#5�  T          @��\�|�Ϳ�Q�@p  B)33CD���|���G�@XQ�B�CO
                                    Bxj#DR  "          @�  �x�ÿ���@n{B*33CD8R�x�ÿ��H@W
=Bz�CN�q                                    Bxj#R�  "          @�G��|�Ϳ}p�@p  B*z�CB��|�Ϳ�=q@Z�HB�CL޸                                    Bxj#a�  
�          @�ff��zῇ�@qG�B%CBff��z��z�@Z�HB�RCL��                                    Bxj#pD  
�          @�
=�����=q@qG�B%(�CB�������@Z�HB  CL��                                    Bxj#~�  
�          @��\��33���@hQ�B"��CB�=��33���@R�\B�CL��                                    Bxj#��  
�          @����w��s33@\��B#�CA޸�w���p�@H��B{CL
                                    Bxj#�6  
�          @�G��x�ÿ5@`��B&z�C>\)�x�ÿ�G�@P  B��CI.                                    Bxj#��  �          @���u��(�@Z=qB%�C:s3�u��(�@N{B��CE��                                    Bxj#��  �          @�\)�z=q��=q@^{B&{C7�R�z=q����@S�
B  CCQ�                                    Bxj#�(  
�          @���y���8Q�@XQ�B#p�C6���y���u@O\)B��CA�)                                    Bxj#��  
�          @��\�tz����@UB$��C5�\�tz�aG�@N{BC@�3                                    Bxj#�t  
�          @����r�\�L��@S33B$Q�C4��r�\�Q�@L��B(�C@33                                    Bxj#�  
�          @����u�=�G�@O\)B C2O\�u��&ff@J�HB  C=�\                                    Bxj$�  
�          @�Q��tz�=L��@N�RB �C3G��tz�5@J=qBG�C>}q                                    Bxj$f  �          @�\)�qG����
@P  B#{C5#��qG��Tz�@I��B��C@u�                                    Bxj$   �          @����u���Q�@P  B!ffC5^��u��Y��@H��B��C@�                                    Bxj$.�  
�          @�p��h�þ�Q�@R�\B'��C9�{�h�ÿ�\)@G
=B�CE                                    Bxj$=X  
�          @��o\)���R@L��B!�RC8���o\)��ff@B�\B
=CC�                                    Bxj$K�  �          @���mp���Q�@L(�B"p�C9xR�mp����@AG�B
=CDp�                                    Bxj$Z�  
�          @��
�mp��!G�@EB\)C=�{�mp���=q@6ffBz�CG                                    Bxj$iJ  T          @�\)�^{�aG�@EB#Q�CB33�^{��=q@333B(�CL}q                                    Bxj$w�  "          @����Z=q��z�@J=qB%Q�CF�=�Z=q��\)@333B33CP�                                    Bxj$��  �          @��
�XQ����@G�B  CMY��XQ���\@)��B�
CV)                                    Bxj$�<  �          @��H�R�\��@Q�B)�
CKY��R�\�	��@6ffB��CU33                                    Bxj$��  T          @��H�HQ���@S33B,ffCO�q�HQ��Q�@4z�B�HCYB�                                    Bxj$��  �          @���C33��=q@W
=B1Q�COp��C33��@8��B��CYff                                    Bxj$�.  �          @���I���У�@Tz�B,�\COO\�I����@5BG�CX��                                    Bxj$��  
�          @�{�J=q��@Z=qB.p�CO���J=q��@:=qB��CY�
                                    Bxj$�z  �          @�z��E����@VffB-
=CR+��E��!�@5�B�\C[s3                                    Bxj$�   �          @�
=�AG����@b�\B7�CPh��AG���@C33B�CZǮ                                    Bxj$��  �          @��
��ff�L��@b�\B/�Cp����ff�|(�@+�A��Cuu�                                    Bxj%
l  "          @�p����]p�@`��B-  Cw�ÿ���{@%A���C{T{                                    Bxj%            @�
=�������@FffB�\C�@ ������\)@�A�p�C��)                                    Bxj%'�  �          @��?������@+�A�C�?������
?���A��
C��                                    Bxj%6^  �          @�33?�z���
=@/\)A�G�C��?�z���  ?�33A��C���                                    Bxj%E  �          @��@y���r�\?�=qA[�C��f@y����  >\@{�C�"�                                    Bxj%S�  �          @���@q��k�?�A��C��H@q���G�?z�HA!p�C���                                    Bxj%bP  �          @�Q�@8���|��@=p�A�\)C�
@8����G�?�
=A���C�8R                                    Bxj%p�  T          @��@�R���@6ffA�z�C���@�R��p�?�(�A��\C�e                                    Bxj%�  
�          @�p�@&ff��Q�@AG�B33C�z�@&ff���
?�(�A��C���                                    Bxj%�B  �          @��@��(�@J�HB	�RC�o\@����@A��RC��
                                    Bxj%��  �          @�  ?�p�����@P��B
�C�ff?�p���{@��A���C���                                    Bxj%��  �          @��?ٙ�����@S33BC��
?ٙ���ff@�A���C�|)                                    Bxj%�4  �          @�
=?�=q����@Z=qBG�C�g�?�=q���H@  A���C�P�                                    Bxj%��  �          @�
=?�G����@R�\B�C�o\?�G���=q@Q�A�G�C�E                                    Bxj%׀  �          @�{?�ff��z�@Mp�B
z�C��3?�ff��G�@�
A��\C���                                    Bxj%�&  �          @��R������\@mp�B"Q�C�f�����33@#33Aԏ\C�+�                                    Bxj%��  T          @���{��(�@eB�
C����{���
@�HA�C�
                                    Bxj&r  �          @��H�k���{@p��B �RC����k���
=@$z�A�33C��                                     Bxj&  *          @�{�5�{�@���B4C�ٚ�5����@<(�A�z�C�                                    Bxj& �            @�ff���
����@o\)B$
=C��þ��
���@%�Aי�C�+�                                    Bxj&/d  �          @��J=q�n{@�p�B>p�C�  �J=q��(�@H��B  C�&f                                    Bxj&>
  �          @�z�Q��`��@���BH33C�uÿQ���ff@Tz�B�HC��\                                    Bxj&L�  �          @���+����\@qG�B)ffC�P��+���(�@*=qA��HC��                                    Bxj&[V  �          @�p��W
=���
@K�Bz�C��R�W
=��Q�?���A�G�C�u�                                    Bxj&i�  �          @���p����z�@Z�HB{C��\�p����33@\)A��HC��                                     Bxj&x�  �          @�����H���R@J=qB	��C~ٚ���H��33?�(�A�
=C�W
                                    Bxj&�H  �          @������z�@]p�Bp�C{)�����@A�ffC}                                    Bxj&��  �          @��H�������\@c33B�\C|\������=q@(�A��
C~�3                                    Bxj&��  �          @��\��ff����@^�RB��C|
=��ff��Q�@Q�A�=qC~��                                    Bxj&�:  T          @���������@.{A��C��R������?��HAu�C�'�                                    Bxj&��  �          @���=�G����\@*�HA�C��=�G����H?�33Aj�\C��q                                    Bxj&І  T          @�=q<����@!�A�ffC�*=<���z�?��RAO�C�&f                                    Bxj&�,  �          @�
=�u��{@�Aď\C����u���?�G�A#�C��\                                    Bxj&��  �          @���\)���?���A�
=C�33�\)��G�?\)@���C�>�                                    Bxj&�x  �          @��þL����z�@��A�Q�C��{�L������?xQ�A&�RC��                                    Bxj'  �          @��R=��
��>W
=@�RC�j==��
��녿�33�B�HC�l�                                    Bxj'�  �          @��\������@aG�B�\C{�Ϳ�����@�Aƣ�C~ff                                    Bxj'(j  �          @�  �3�
�E�@uB*��Cc�)�3�
�|(�@<��A���Cj}q                                    Bxj'7  �          @�{���H��{@l��B��Ct�ÿ��H��\)@"�\AʸRCx��                                    Bxj'E�  �          @��׿�\���@,��A���Cy�=��\��Q�?�
=Af�HC{n                                    Bxj'T\  �          @���.{����?���A��C�%�.{��(�?!G�@�33C�`                                     Bxj'c  �          @�{�aG����H?ǮA�Q�C�˅�aG����>.{?�\C��R                                    Bxj'q�  �          @��?Y���XQ�@�{BOp�C��?Y����z�@\��B�\C�z�                                    Bxj'�N  �          @�zᾮ{�<��@���Bb�
C����{�\)@hQ�B(�
C���                                    Bxj'��  T          @�G�?p���Tz�@��BT��C��?p����(�@g�B�
C��                                    Bxj'��  �          @��
@G�����@n{BQ�C��{@G����\@!G�A�G�C�                                    Bxj'�@  �          @�(�@*=q��  @G�A���C�G�@*=q��z�?��A��C���                                    Bxj'��  �          @�
=@c33���?�z�A�(�C��{@c33��ff?=p�@�ffC���                                    Bxj'Ɍ  �          @�\)@B�\���H@ ��A�ffC�Ǯ@B�\��ff?G�@���C���                                    Bxj'�2  �          @���@aG���������<��C���@aG��G����R�R�C��                                     Bxj'��  �          @Å@y���O\)�R�\��HC�  @y���33��Q��&{C��)                                    Bxj'�~  �          @��@qG��b�\�,(���p�C�b�@qG��/\)�`  ��\C���                                    Bxj($  �          @���@W����
=��
?h��C��=@W���  �xQ��'�
C��                                    Bxj(�  �          @ƸR@�R���\@Q�B ��C�:�@�R����@   A�\)C��                                     Bxj(!p  �          @�=q@qG���Q�@��A�=qC���@qG���?xQ�A�HC��)                                    Bxj(0  �          @�Q�@E���33@B�\A�
=C�4{@E���
=?�Q�AqC���                                    Bxj(>�  �          @�  @~�R��{@�A���C��@~�R��p�?��HA-G�C�|)                                    Bxj(Mb  
�          @�G�@�����G�@'�A�(�C���@�����=q?�
=AM�C���                                    Bxj(\  �          @Ӆ@p������@ ��A��C��@p����Q�?���A(Q�C�Ǯ                                    Bxj(j�  �          @�Q�@_\)��  @.�RAƸRC�&f@_\)��G�?�z�AH��C���                                    Bxj(yT  �          @�Q�@^�R��=q@&ffA��C��@^�R���\?��\A4Q�C��\                                    Bxj(��  �          @�@N�R���H@EA���C���@N�R��\)?�ffA��\C���                                    Bxj(��  �          @���@�������?��HA|Q�C��{@������\?��@��RC��q                                    Bxj(�F  �          @��@�  ����?333@�C�B�@�  ��녾�(��~�RC�%                                    Bxj(��  �          @���?�{���@FffBz�C��q?�{��Q�?��A�=qC��H                                    Bxj(  �          @�  ��������@o\)B�Cv�׿������H@�A�
=Cy�f                                    Bxj(�8  �          @��@ ����  @�(�B$�C�
=@ ����{@<(�A�p�C�u�                                    Bxj(��  �          @�p�>�(���33@���BEffC��R>�(����R@c�
B	
=C�W
                                    Bxj(�  �          @�{>����@�G�B(�C��>���{@9��A�G�C�g�                                    Bxj(�*  �          @�  >�Q���ff@�
=B-�RC�*=>�Q����R@E�A�{C���                                    Bxj)�  
�          @У�?(����R@�{BE{C�&f?(����@j=qB�C�E                                    Bxj)v  �          @У�>��R�u�@�  BW(�C�T{>��R���\@��BG�C�                                    Bxj))  �          @ҏ\��\)�n�R@���B]�C��쾏\)����@�\)B �C�aH                                    Bxj)7�  �          @�녿z�����@���BF33C��H�z�����@b�\B	z�C���                                    Bxj)Fh  �          @�
=>���~{@�{BG
=C��>����=q@^�RB	�RC��)                                    Bxj)U  �          @�ff?
=�A�@��Bu��C�|)?
=��p�@��B8�C�Ǯ                                    Bxj)c�  �          @�{>��R�5�@�z�Bxz�C��>��R���@�{B;(�C��                                    Bxj)rZ  �          @�=q@
=�xQ�@��B2�C�B�@
=��z�@J�HA��
C��                                    Bxj)�   �          @�=q?���=q@6ffA�z�C��3?����?��
AyG�C��                                    Bxj)��  �          @�(�@u���=q���H��
C��
@u��|���7���=qC��                                    Bxj)�L  �          @��@~�R���R������C�  @~�R�~{�S33��
=C��                                    Bxj)��  �          @�Q�@��H��ff�����C��@��H��
=�U���\C��                                    Bxj)��  �          @�{@`  ��  �/\)��33C��@`  �fff�u�C��                                    Bxj)�>  S          @�Q�@u���\)�u��C��=@u���G��G����C�
                                    Bxj)��  �          @�Q�@�ff���?=p�@��HC��
@�ff��{�
=q��  C��H                                    Bxj)�  �          @�  @�G�����?���A33C���@�G����;#�
����C�&f                                    Bxj)�0  �          @љ�@��\����?���A��C�Ǯ@��\��(�?\)@��C�Ǯ                                    Bxj*�  �          @Ӆ@�ff��z�@	��A�\)C�\@�ff����?O\)@���C�޸                                    Bxj*|  �          @ҏ\@x����p�@   A��RC���@x����p�?��A�
C��                                     Bxj*""  �          @���@z�H��ff@@��A�
=C��R@z�H���H?�Q�Ak
=C��                                    Bxj*0�  �          @׮@k�����@>{A�(�C��@k�����?ǮAW\)C��                                    Bxj*?n  �          @�G�@W����
@:=qA�
=C���@W���
=?�33A>�HC�AH                                    Bxj*N  �          @љ�@Z=q��\)@=p�A�\)C�� @Z=q���?ǮA\Q�C�8R                                    Bxj*\�  �          @�p�?�����=q?���A��C���?�����33>#�
?��HC�Q�                                    Bxj*k`  �          @��@��H�b�\@-p�A�=qC�g�@��H����?У�As�C�*=                                    Bxj*z  T          @ʏ\@��׿Ǯ@;�A�RC��H@����33@��A��RC�0�                                    Bxj*��  �          @�p�@�
=��p�@5�A�(�C�8R@�
=�(�@  A�z�C��R                                    Bxj*�R  �          @�@�=q�z�@.�RA�33C�:�@�=q�>�R?�p�A�  C�aH                                    Bxj*��  �          @��
@�G�����@ ��A���C���@�G��?�  AR�RC�u�                                    Bxj*��  �          @��@�p��J�H@
�HA�33C�AH@�p��h��?��HA,��C�k�                                    Bxj*�D  �          @Ӆ@�z��R�\?���A^�HC�N@�z��fff?��@�  C�&f                                    Bxj*��  �          @��@����.{?!G�@���C��{@����2�\�����=qC���                                    Bxj*��  �          @�(�@ʏ\�)��>�{@3�
C��@ʏ\�(�þ���\(�C��\                                    Bxj*�6  �          @���@��H�,��=�Q�?@  C�z�@��H�'��(������C��=                                    Bxj*��  �          @���@�
=��ýL�;�(�C��R@�
=�녿8Q���
=C�E                                    Bxj+�  �          @�Q�@�p��33��Q��C33C�'�@�p���\)�k����C��f                                    Bxj+(  �          @�Q�@�\)�H�þaG���=qC�%@�\)�=p�����  C�ٚ                                    Bxj+)�  �          @ٙ�@���w�    <#�
C��{@���n{��=q�\)C�                                      Bxj+8t  �          @�G�@�=q�{��u�   C���@�=q�p�׿�z��z�C��{                                    Bxj+G  �          @��@�G���G�>�p�@A�C�q@�G���
=�Y����
=C�W
                                    Bxj+U�  �          @߮@�p���{>�z�@=qC��@�p����\������C�9�                                    Bxj+df  �          @�Q쿱����@J�HA�Q�C�������G�?��
A]��C��=                                    Bxj+s  �          @��
�a���ff@l��A��\Cju��a���G�@	��A��\Cn�)                                    Bxj+��  �          @�z��P�����@l��A�  Cm�3�P�����@A�(�Cqu�                                    Bxj+�X  �          @��HQ���=q@i��A��HCo�=�HQ���z�?��RA��HCr�q                                    Bxj+��  �          @�33�@����33@c33A�Q�Cp�H�@����z�?��Av=qCs�H                                    Bxj+��  �          @�  �G���Q�@Y��A�ffCoG��G���Q�?�G�Ah��Cr�                                    Bxj+�J  �          @���C�
��p�@g�A���Co\)�C�
���@   A�=qCr�                                    Bxj+��  �          @��G
=��{@l��A�p�Co��G
=��G�@�
A�
=Cr�3                                    Bxj+ٖ  �          @��A���ff@Y��A��HCp��A���{?ٙ�A\��Cs��                                    Bxj+�<  
�          @�z��333��p�@n{A��Cr���333��Q�@ ��A���Cu�                                    Bxj+��  T          @����AG���33@j=qA��RCp�\�AG���p�?�(�A�{Cs��                                    Bxj,�  �          @�(��P  ���@l(�A�p�Cm�
�P  ��Q�@33A�p�Cq�)                                    Bxj,.  �          @��
�N�R����@l��A���Cm�H�N�R��  @�
A�z�Cq�                                    Bxj,"�  �          @���[�����@fffA��HCk���[����H?��RA�
=Co�\                                    Bxj,1z  �          @��H�Z�H��p�@Z=qA��Cl� �Z�H��{?�\Ag�Cp
=                                    Bxj,@   �          @�(��P����=q@\��A�p�Cn}q�P�����H?�  Ad  Cq�
                                    Bxj,N�  �          @�G��@  ��
=@P  A�Q�CqG��@  ��?\AG�Ct!H                                    Bxj,]l  �          @��H�+���  @e�A�=qCt��+����?���AmG�Cv��                                    Bxj,l  �          @�(��+���ff@n�RA�z�Cs���+����?�p�A���Cv�3                                    Bxj,z�  �          @��
�9����z�@}p�BQ�Cp���9���\@�A��Ctu�                                    Bxj,�^  �          @����5����@|(�B��Cq�{�5����@�RA��CuL�                                    Bxj,�  
�          @�\�"�\���\@xQ�BG�Ct���"�\��\)@��A�\)Cw�)                                    Bxj,��  
�          @�� ������@|��B�Cz�� ���θR@��A�Q�C|�                                     Bxj,�P  �          @���
��=q@q�B =qCw����
��{?�(�A�Q�CzJ=                                    Bxj,��  �          @�p��������@fffA�{Cy��������H?�p�A`(�C{��                                    Bxj,Ҝ  �          @�(��R�\�hQ�@�{B4�Cc�R�\��p�@h��A��RCl:�                                    Bxj,�B  �          @������@�B-��Cr(������@UA�{Cws3                                    Bxj,��  �          @�R�����\@���B+�CvL������@P  A�z�Cz�                                     Bxj,��  �          @�Q������G�@��RB,��Ct���������@S�
A�p�Cyz�                                    Bxj-4  �          @�Q��%����H@��B-�Cp���%����
@Y��A�\)Cv@                                     Bxj-�  �          @�=q�,�����@�G�B.ffCo\)�,����33@\��A�G�Cu@                                     Bxj-*�  �          @�33�   ��{@��\B.��Cq�q�   ��  @\(�A�G�Cwh�                                    Bxj-9&  �          @�ff�(Q���33@�\)B2�\CpB��(Q����R@g
=A�{Cv33                                    Bxj-G�  �          @�
=�2�\���
@��
B8(�CmaH�2�\����@s�
A�Ct33                                    Bxj-Vr  �          @�Q��(���G�@�33B2��CqǮ�(����@_\)A��Cwp�                                    Bxj-e  
�          @���(���@��B2�RCt���(���Q�@]p�A�p�Cy�q                                    Bxj-s�  �          @�Q��p���p�@�33B2z�Ct���p����@\��A���Cy�q                                    Bxj-�d  �          @�\)�p�����@��B2  Ct�=�p����R@Z�HA��Cy�H                                    Bxj-�
  �          @�{�ff���@��B5(�Cu}q�ff��{@^�RA�
=Cz��                                    Bxj-��  �          @�{�ff��{@�G�B1��Cu޸�ff��  @XQ�A�Cz�3                                    Bxj-�V  �          @���\��p�@�33B,  Cs޸��\��@L��A�Cx�)                                    Bxj-��  �          @�ff�.{����@�p�B!�CpY��.{��
=@?\)A�(�Cu��                                    Bxj-ˢ  �          @���.�R���@���B��Co���.�R��=q@7�A���Ct��                                    Bxj-�H  �          @��
��
��{@�=qB&�
CvY���
���
@:=qAȣ�Cz��                                    Bxj-��  �          @�ff�/\)����@�B,(�Cn���/\)���@S�
A��Ct��                                    Bxj-��  �          @�p��(����@�33B5Q�Cq��(���Q�@_\)A�p�Cw{                                    Bxj.:  �          @�p��33��ff@��\B=��Cs��33���@b�\A���Cy��                                    Bxj.�  �          @޸R�e����@���B
=Cf��e���{@p�A���Cl�H                                    Bxj.#�  �          @��S33��\)@�(�B�Ch��S33��(�@7
=A�
=Cnp�                                    Bxj.2,  �          @߮�B�\��
=@��B'p�CjG��B�\���R@G�A�z�Cp��                                    Bxj.@�  T          @ᙚ�Mp����\@��B �
Ciu��Mp�����@?\)A�Q�Co�H                                    Bxj.Ox  �          @ڏ\�0  ��
=@��B(��Cl�f�0  ��@AG�A�33Cs�                                    Bxj.^  �          @���2�\���@�
=B'�HCk���2�\����@=p�Aң�CrE                                    Bxj.l�  �          @�G��=q�|(�@��RB)�CnxR�=q����@1G�A���CtxR                                    Bxj.{j  �          @�����H�z=q@�G�B+�CnJ=��H���@6ffA�p�Cts3                                    Bxj.�  �          @�  ������@��\B3��Cq�f����Q�@EA㙚Cw�=                                    Bxj.��  �          @�=q���\)@�B*{Cq\�����@8Q�A�ffCv�)                                    Bxj.�\  �          @��ÿ����ff@���B!(�C�\������
@+�A�G�C�*=                                    Bxj.�  �          @�׿��
���@�{B'��CzͿ��
��(�@:=qAîC}�\                                    Bxj.Ĩ  �          @�=q�������@���B  Cy�H�������H@�RA�Q�C}\                                    Bxj.�N  �          @��H��  ���@�33B��C}����  ��p�@!G�A�\)C�.                                    Bxj.��  T          @�Q쿋���33@�z�B�C�=q�����
=@{A���C�7
                                    Bxj.�  �          @�\���H����@�Q�B�HC�Ф���H���H@�A�z�C��=                                    Bxj.�@  �          @�׿�����{@���B  C�Ǯ�����У�@z�A�z�C���                                    Bxj/�  �          @�ff��\)��  @��B�HC�C׿�\)�У�@��A��C�%                                    Bxj/�  �          @ۅ������z�@�33B��C�e������p�@
=qA��HC�Ff                                    Bxj/+2  �          @�
=�����@��B�C�9�����Ǯ@�A�z�C�/\                                    Bxj/9�  �          @أ׿����  @�{B$
=CxLͿ����p�@,(�A���C|O\                                    Bxj/H~  �          @�\)>�(���\)@U�A�\)C�9�>�(���Q�?��AG�C��3                                    Bxj/W$  �          @љ�?5��{@Dz�A��C���?5���
?���A�
C�+�                                    Bxj/e�  �          @޸R>\����@]p�A��HC��>\��=q?��A1G�C��R                                    Bxj/tp  �          @�(�?&ff��G�@L��A��C�{?&ff��  ?���A�
C�                                    Bxj/�  �          @�=q?Tz���@P��A�RC���?Tz����?�AC��=                                    Bxj/��  T          @�G�?W
=��@K�A��
C�
=?W
=��z�?��A�C��q                                    Bxj/�b  �          @أ�?�������@#33A��C���?����љ�>�G�@p  C�<)                                    Bxj/�  �          @�=q?�����33@,��A���C�>�?�����z�?�@���C���                                    Bxj/��  �          @ڏ\?�����@Q�A�G�C��=?�����?�  A*=qC���                                    Bxj/�T  �          @���@�R��\)@P  A�  C�{@�R�Ǯ?��
A.ffC��R                                    Bxj/��  �          @أ�?�(�����@,(�A�33C�ٚ?�(��ҏ\?\)@���C�L�                                    Bxj/�  �          @���?�\)��G�@z�A�G�C�9�?�\)��ff>�?���C���                                    Bxj/�F  �          @��?�p��Ǯ@��A��HC�?�p���>aG�?�C�E                                    Bxj0�  �          @�z�?�33��33@.�RA���C���?�33���?\)@�ffC���                                    Bxj0�  �          @���@   ����@-p�A���C�7
@   ��=q?�@��C�z�                                    Bxj0$8  �          @��@)�����@{A�  C��\@)����(�=�?z�HC�C�                                    Bxj02�  �          @�ff@N�R���H?�33A~ffC�t{@N�R��z�\)��C���                                    Bxj0A�  �          @޸R@:=q����@#�
A�
=C�aH@:=q�ə�?   @�z�C�g�                                    Bxj0P*  �          @޸R@�H��Q�@AG�A�ffC�e@�H��{?n{@�\)C�N                                    Bxj0^�  }          @�ff@��@Z�HA�(�C�R@�Ϯ?��A1��C���                                    Bxj0mv  �          @�\)?�z���p�@fffA�=qC�S3?�z��љ�?�  AF�\C�&f                                    Bxj0|  �          @�
=@  ��Q�@N{A�(�C��=@  ��  ?�{A�\C���                                    Bxj0��  �          @߮@z�����@K�A��HC��\@z���Q�?���Az�C��                                    Bxj0�h  �          @�Q�@z���33@Dz�A�(�C���@z��љ�?n{@�(�C���                                    Bxj0�  �          @���@�
����@C33A͙�C��3@�
�ҏ\?c�
@�  C��                                    Bxj0��  �          @�=q@z���p�@Dz�A�Q�C��3@z��Ӆ?fff@陚C���                                    Bxj0�Z  �          @�=q@p����R@7�A�=qC�7
@p��ҏ\?0��@�z�C�@                                     Bxj0�   �          @�G�@{��33@C33A��C�n@{��G�?c�
@�  C�U�                                    Bxj0�  �          @��@#�
��33@9��A�\)C���@#�
�Ϯ?@  @�33C��H                                    Bxj0�L  �          @߮@%��33@3�
A�  C��{@%�θR?(��@�C��                                    Bxj0��  �          @�
=@#�
���\@6ffA��RC���@#�
��ff?0��@��RC��\                                    Bxj1�  �          @�G�@!���G�@G�A�{C�˅@!���  ?u@��C��q                                    Bxj1>  T          @��@"�\���@@  A�33C���@"�\��G�?Q�@���C���                                    Bxj1+�  �          @�\)@1����\@(Q�A�\)C��q@1���(�>�@{�C��                                    Bxj1:�  �          @�Q�@;�����@'�A���C�e@;����H>��@w
=C�c�                                    Bxj1I0  �          @�  @<�����H@�RA�=qC�g�@<�����H>���@+�C�|)                                    Bxj1W�  �          @�G�@>{��ff@z�A�  C�Ff@>{��(�>�?���C�|)                                    Bxj1f|  �          @�\@E���\)@p�A�G�C��H@E����
<#�
=�Q�C��f                                    Bxj1u"  �          @��
@E���\)@�A���C��q@E���p�=�?�  C��\                                    Bxj1��  �          @�33@Dz���  @\)A�G�C��=@Dz�����<�>uC�˅                                    Bxj1�n  �          @�=q@6ff����@A��RC���@6ff�θR=�G�?^�RC��=                                    Bxj1�  �          @��@0���Å@   A���C�,�@0����33>aG�?�ffC�]q                                    Bxj1��  �          @�R@<(���33@�RA���C��@<(��ҏ\>L��?���C��                                    Bxj1�`  �          @�R@@  �\@{A�{C�&f@@  ���>L��?ǮC�O\                                    Bxj1�  T          @��@2�\��\)@0  A��HC�}q@2�\��=q>��H@z=qC��                                     Bxj1۬  �          @��@)����G�@0��A��C��R@)����(�>��@qG�C��                                    Bxj1�R  �          @ᙚ@�H��{@:�HA�{C��@�H�ҏ\?(��@�33C��                                    Bxj1��  �          @���@����\@Q�A�ffC�>�@��Ӆ?��
A{C��                                    Bxj2�  �          @�Q�@Q���G�@J=qA�z�C�33@Q�����?n{@��HC��                                    Bxj2D  �          @�\)@����=q@`��A�ffC�\@����ff?���A/�C���                                    Bxj2$�  �          @�z�?����R@w
=B\)C���?���
=?�Q�Ab�RC�=q                                    Bxj23�  �          @��
?�  ����@k�B�RC��=?�  ��
=?��RAHz�C���                                    Bxj2B6  T          @�33?�G����
@`  A��
C��3?�G���  ?��
A,��C���                                    Bxj2P�  �          @��?�z���(�@S�
A�RC�b�?�z���{?��A(�C�G�                                    Bxj2_�  �          @�G�@����\@Q�A��C��@���(�?�=qAffC���                                    Bxj2n(  �          @ۅ@����=q@XQ�A�=qC���@�����?�AC�@                                     Bxj2|�  �          @���@�
����@Z=qA��C�XR@�
��(�?��HA"{C��{                                    Bxj2�t  �          @�p�@�����\@QG�A�p�C��)@����(�?�ffAQ�C���                                    Bxj2�  �          @�p�@�����@XQ�A��
C�Ff@��Ϯ?���A=qC��                                    Bxj2��  T          @�Q�@���Q�@\(�A�G�C��3@��Ӆ?�33A�\C��f                                    Bxj2�f  �          @ᙚ?�������@a�A�ffC�U�?������?�(�A�HC�+�                                    Bxj2�  �          @�=q?�z���z�@Z�HA�C��
?�z���\)?���A�C���                                    Bxj2Բ  �          @���@   ����@Q�Aڣ�C�4{@   �ٙ�?c�
@��
C�7
                                    Bxj2�X  �          @�
=@	���\@O\)Aՙ�C�� @	�����H?Tz�@ҏ\C��)                                    Bxj2��  �          @�?�\��@Tz�AڸRC���?�\��
=?\(�@�=qC�q                                    Bxj3 �  �          @�  ?�  ����@I��AΣ�C��?�  �߮?(��@�  C�H                                    Bxj3J  �          @�{@���  @5A�Q�C�^�@���33>\@@  C��R                                    Bxj3�  �          @�
=@G��ƸR@8��A��HC�@G����H>�(�@[�C�33                                    Bxj3,�  T          @�=q?��R��33@C33A�G�C��\?��R���?�@�(�C���                                    Bxj3;<  �          @�\)@%��(�@%A���C�@%��(�=��
?!G�C�P�                                    Bxj3I�  �          @�p�@�G���33=�?c�
C��@�G�������33C��=                                    Bxj3X�  �          @��
@�z�����>���@�C�T{@�z���33����mG�C�                                    Bxj3g.  �          @陚@���������
�&ffC��f@������p�����C��q                                    Bxj3u�  �          @���@�p�����>�{@*=qC�j=@�p���������h��C��                                    Bxj3�z  �          @�@�=q�Å?z�H@�C�� @�=q��G������$Q�C���                                    Bxj3�   �          @��H@��H��  ?
=q@�=qC���@��H���ÿ�(��Q��C�=q                                    Bxj3��  �          @�33@�Q���(�>Ǯ@=p�C�~�@�Q�������]p�C�(�                                    Bxj3�l  �          @��@�(�����>���@p�C���@�(������Q��k�C�t{                                    Bxj3�  �          @�{@��R��ff?�=qA{C�+�@��R�������z�C�9�                                    Bxj3͸  �          @�p�@��\���?B�\@�C���@��\���׿�  �3�
C��3                                    Bxj3�^  �          @�{@�{���>�
=@I��C�3@�{��33����W�
C���                                    Bxj3�  �          @��@����z�>���@p�C���@�����H��33�e��C���                                    Bxj3��  �          @�
=@����{?+�@�  C���@����Q�����?�C�3                                    Bxj4P  �          @�\)@�z���
=>aG�?�{C���@�z����
��\�tz�C���                                    Bxj4�  �          @���@�G��������C�,�@�G������{��C���                                    Bxj4%�  �          @��@��\��>�?n{C���@��\����������\C���                                    Bxj44B  �          @��@�\)���׿s33��C��@�\)����C33����C�*=                                    Bxj4B�  �          @�(�@�p�����L�Ϳ�p�C���@�p���33������HC�H                                    Bxj4Q�  �          @�@�ff��p�>aG�?ǮC���@�ff��=q��\�o�C���                                    Bxj4`4  �          @�{@��R���>Ǯ@6ffC�k�@��R��33��(��f{C�#�                                    Bxj4n�  T          @�@��
�Ӆ���p��C��
@��
��=q�(Q����HC��q                                    Bxj4}�  �          @�(�@vff�ۅ>W
=?ǮC���@vff���=q��ffC�xR                                    Bxj4�&  �          @�(�@�=q��=q>��@[�C���@�=q��  ��
�s
=C�O\                                    Bxj4��  �          @��
@�z����H>\)?��
C�R@�z������\��G�C��                                    Bxj4�r  T          @�ff@�����#�
��  C�xR@����{�����(�C��q                                    Bxj4�  �          @��R@�33��\)?�A�
C��H@�33��  ��{� ��C�y�                                    Bxj4ƾ  �          @��R@�(���G�?8Q�@��C�'�@�(������33�?�
C���                                    Bxj4�d  �          @�\)@�ff��G�<��
>.{C�XR@�ff���H�33��  C�t{                                    Bxj4�
  �          @�{@������R?n{@�ffC�ff@��������Q��'
=C��                                    Bxj4�  �          @�@��R���?�G�AN{C��)@��R��z�&ff��ffC�w
                                    Bxj5V  �          @�@�\)��G�?�  A�C�@�\)��녿�
=�z�C���                                    Bxj5�  �          @�p�@�33�Å?�ffA33C�7
@�33��z῔z��33C�'�                                    Bxj5�  �          @�p�@�����p�?��RA  C���@�����p���  ���C��3                                    Bxj5-H  �          @��@�G���\)?O\)@��
C���@�G��������C33C�,�                                    Bxj5;�  �          @�=q@�G���  ?���A"�\C�K�@�G���=q�����=qC�&f                                    Bxj5J�  �          @���@��H��{>�\)@C���@��H��33���H�mG�C��=                                    Bxj5Y:  �          @��@����33�Tz����C��@�����\�<�����HC�(�                                    Bxj5g�  T          @�  @��������tz�C��@����
=�8Q�����C�7
                                    Bxj5v�  �          @�z�@�z����H�!G���=qC��@�z����H�C�
���\C��R                                    Bxj5�,  �          @�(�@�����ͿTz��\C�.@�����\�J�H���C�1�                                    Bxj5��  �          @�33@���Å�=p�����C�B�@�����\�E�����C�0�                                    Bxj5�x  �          @�33@�ff�����&ff����C��@�ff����>�R��ffC��H                                    Bxj5�  �          @���@����˅�!G���(�C�XR@�����33�E�����C��                                    Bxj5��  �          @��R@�p����H�5��  C���@�p���=q�C33��ffC��                                    Bxj5�j  �          @�@�=q��ff������RC���@�=q��\)�9����33C�U�                                    Bxj5�  �          @�@��R��ff���`  C�33@��R�����1���p�C��H                                    Bxj5�  �          @�33@�����Ϳ(����\C�c�@�����8����33C�<)                                    Bxj5�\  �          @��@��R���R��ff��RC�k�@��R��ff�c�
��RC��                                    Bxj6	  �          @��@�
=��=q���H�0z�C���@�
=��Q��i����RC�]q                                    Bxj6�  �          @��\@����G���z��&�\C��@����\)�l����(�C���                                    Bxj6&N  �          @��\@�Q���z�
=���C��{@�Q����R�,(���
=C���                                    Bxj64�  �          @�33@����
=������C�S3@����{�����C��                                    Bxj6C�  �          @��@������׾��aG�C�!H@������������C��\                                    Bxj6R@  �          @��@�����G������:�HC�@�����\)����  C��                                     Bxj6`�  �          @���@�{���׿G���  C��R@�{��Q��5���=qC���                                    Bxj6o�  �          @��@�p���(�������C���@�p����R�'
=��
=C���                                    Bxj6~2  �          @���@�p����Ϳ.{��
=C�z�@�p���{�,����=qC��                                    Bxj6��  �          @�ff@����p������ (�C�K�@����p��U�ǅC�C�                                    Bxj6�~  �          A   @�����z�\�0(�C�xR@������H�^{��=qC��                                    Bxj6�$  �          A (�@�����
=��G��K
=C�c�@��������z�H����C���                                    Bxj6��  �          A ��@������Q��B�RC���@����R�|(����HC��)                                    Bxj6�p  �          A ��@�����{���j{C�n@�����(�������
C�*=                                    Bxj6�  �          A{@�=q�Å��G��G�C���@�=q���
���H��\)C��                                    Bxj6�  
�          A33@�33���
���Lz�C��f@�33�����z���C�33                                    Bxj6�b  �          A�H@�Q�����z���p�C�n@�Q����
�q���G�C���                                    Bxj7  �          AQ�@��
�׮�fff��33C��q@��
�����e�����C�                                    Bxj7�  �          A	@��R��z�k����
C��3@��R��
=�:�H��
=C�8R                                    Bxj7T  �          A\)@�33���!G���p�C��@�33��G��`������C��3                                    Bxj7-�  �          A��@�p���
=?\(�@�33C�ff@�p����׿�\�9�C���                                    Bxj7<�  �          A��@�z����?k�@�{C�q@�z���(���  �7
=C���                                    Bxj7KF  �          A�@�ff��
=>�@B�\C��@�ff���H��R�e�C��q                                    Bxj7Y�  �          Az�@Ӆ�θR?Tz�@��
C��
@Ӆ��
=��z��=��C�\)                                    Bxj7h�  �          Ap�@ᙚ����?�ff@ȣ�C��@ᙚ��Q��Q��"=qC�4{                                    Bxj7w8  �          A�H@��H����?8Q�@���C���@��H�Ǯ��\�C
=C�T{                                    Bxj7��  T          A
=@�����Q�?��Az�C��
@�����  ��
=��C���                                    Bxj7��  �          A��@�33��=q?���A=qC���@�33���Ϳ��\��C�s3                                    Bxj7�*  �          Az�@��H��\)?�(�A!�C��R@��H��(����\���C��H                                    Bxj7��  �          A  @�\)�\@�AO�C��R@�\)��{����O\)C�+�                                    Bxj7�v  �          A@ᙚ��  @ffAc�C�Ф@ᙚ����Q���C��{                                    Bxj7�  �          A��@�
=��Q�?��HA%p�C��@�
=���Ϳ���˅C��R                                    Bxj7��  �          AQ�@�
=��z�@N�RA�\)C��@�
=��
=?z�@[�C��                                    Bxj7�h  �          A@�R���\@4z�A�G�C�"�@�R��  >aG�?���C���                                    Bxj7�  �          A��@�z���p�@`��A��C�]q@�z����?��@�{C��
                                    Bxj8	�  �          A��AG��\��@��\AѮC�p�AG���ff@�RAj�RC�C�                                    Bxj8Z  �          A��A Q��~�R@y��A�=qC�˅A Q���  ?�\)A0(�C�c�                                    Bxj8'   �          A33@�����@
�HAO33C�|)@�����=q��
=�\)C��3                                    Bxj85�  T          Az�@�z����?�  A(��C�ٚ@�z���\)�xQ����\C�s3                                    Bxj8DL  �          A��@���
=?h��@��RC��=@����׿�ff�.{C���                                    Bxj8R�            A�H@�����(����,��C�W
@�������Mp���G�C�33                                    Bxj8a�  T          A
=@�=q�ʏ\>L��?���C��\@�=q���\�   �o
=C��q                                    Bxj8p>  T          A
=@�R����?���@�{C�}q@�R��ff�����RC��{                                    Bxj8~�  �          A(�Aff���\@��Ac�C�]qAff��{>���?���C�Ǯ                                    Bxj8��  �          Az�AG���=q@Q�AJffC���AG���G��#�
�uC�k�                                    Bxj8�0  �          A  A����?���A  C�^�A����5���C��                                    Bxj8��  �          Az�AG���=q?��HA!��C��)AG����H�   �<(�C�K�                                    Bxj8�|  �          A�
A(���G�?�33A3�C��3A(���zᾨ�ÿ�Q�C�                                    Bxj8�"  �          A\)A��(�?�(�A
�HC��RA��Q�^�R���
C���                                    Bxj8��  
�          A�R@�=q��(�?8Q�@�Q�C�  @�=q���Ϳ�G��(  C��3                                    Bxj8�n  �          A�@������?\)@U�C��@�����녿�33�6�\C��R                                    Bxj8�  �          A�H@��
���?8Q�@���C�B�@��
���
��  �'
=C��{                                    Bxj9�  {          A
=@�p���(�?\)@UC�C�@�p������G��@z�C�                                    Bxj9`  �          A�\@�����33?G�@�(�C�'�@�����z���H�#�C���                                    Bxj9   �          Az�AQ�����?�ff@��
C�J=AQ����
��G����C�                                      Bxj9.�  �          A��A33���R?��R@�  C�ǮA33��\)��������C��R                                    Bxj9=R  �          A(�A�R��
=?���@θRC��
A�R�����\��ffC��\                                    Bxj9K�  �          A(�A{��33?�p�@�Q�C��RA{����}p����C��3                                    Bxj9Z�  �          A�RA
=���R?��
@��HC�ffA
=���׿�  ��{C�=q                                    Bxj9iD  �          A�RAp���?�Q�A��C�Y�Ap����H�:�H��33C��\                                    Bxj9w�  �          A�A(����H?�  A%C�}qA(���p����R����C��                                     Bxj9��  �          A33Ap����@�A^ffC�/\Ap���{>L��?�C���                                    Bxj9�6  �          A�A  ���H?�@�{C�!HA  ��p��Tz����HC��=                                    Bxj9��  �          A��A�H���\��{���C��{A�H�w
=�G
=���C�Y�                                    Bxj9��  �          A�A   ��ff��  �(Q�C�"�A   �k��n�R��C��H                                    Bxj9�(  �          AQ�@������\��ff��C��H@����x���g
=����C���                                    Bxj9��  �          Az�@�{�����
=�!�C��=@�{�p���l(���z�C�XR                                    Bxj9�t  �          A�@�
=��Q��Q��"�HC��@�
=�p���n{����C�^�                                    Bxj9�  �          A��@��
���\���1C��R@��
�p  �xQ���  C�B�                                    Bxj9��  �          A��@����녿���6�RC��)@���n{�z�H��Q�C�Y�                                    Bxj:
f  �          A��@��\��=q�33�FffC��f@��\�i�����\��{C�~�                                    Bxj:  �          A��@��\����z��H��C���@��\�hQ����H����C��3                                    Bxj:'�  �          A��@����33��IC�n@���i�����
�̏\C�w
                                    Bxj:6X  �          A��@�G���=q�
�H�R=qC�xR@�G��e��{��=qC���                                    Bxj:D�  �          A��@����z���\�^�RC�
=@���W���
=��Q�C�b�                                    Bxj:S�  �          A��@�G���ff���Mp�C��@�G��|����=q��33C�0�                                    Bxj:bJ  �          A�@���p���\�D��C�Ff@��|����\)��(�C�H�                                    Bxj:p�  �          Ap�@�p���  ��\�]�C��@�p��k���z���Q�C�,�                                    Bxj:�  �          AG�@�ff��=q�&ff�{�C�O\@�ff�X�����\��RC�                                      Bxj:�<  �          A{@��������d  C���@��j�H��
=�ݮC�9�                                    Bxj:��  �          A�@�{��p�� ���@��C�j=@�{�|�����R���C�e                                    Bxj:��  �          A@���{�{�n�HC��
@��b�\������p�C���                                    Bxj:�.  �          Ap�@�
=���R��\�\��C��@�
=�hQ���(���33C�g�                                    Bxj:��  �          Az�@���������b�RC��@���c�
��z��ۮC��=                                    Bxj:�z  �          A��@����  �+�����C�W
@���C33������
=C�Y�                                    Bxj:�   �          Ap�@�����z��0  ��C��=@����:=q��G�����C��                                    Bxj:��  �          A�@����{�  �[�
C�ٚ@���vff��ff���C�C�                                    Bxj;l  �          A\)@�ff����
=q�S�
C��@�ff�vff���
��{C�Q�                                    Bxj;  �          A�@��H�����{�r�RC�*=@��H�Y����\)��RC��\                                    Bxj; �  �          A�A�R��(��(��nffC��RA�R�%��G����
C�8R                                    Bxj;/^  �          A\)A{��ff��H�m��C�U�A{�)�������G�C��
                                    Bxj;>  �          A�HA ����  ��R�s�C��A ���*�H��(���C��3                                    Bxj;L�  �          A�A z����333����C�=qA z��p���z�����C�z�                                    Bxj;[P  �          AQ�@������J�H����C���@���=q������33C�~�                                    Bxj;i�  �          Az�A Q���G��G���G�C�� A Q��������Q�C�aH                                    Bxj;x�  �          AQ�@�\)��(��G
=����C�Q�@�\)�����z���  C�3                                    Bxj;�B  �          A��@�������9�����C��{@����5���ff��Q�C�
=                                    Bxj;��  �          A�
@�������L(���
=C�8R@����{��33��C�.                                    Bxj;��  �          A�@������Mp���ffC��)@��$z���p���ffC���                                    Bxj;�4  �          A\)@�(�����O\)��ffC��\@�(��%���
=��p�C���                                    Bxj;��  �          Aff@�=q��33�I����ffC�^�@�=q�)��������G�C�\)                                    Bxj;Ѐ  �          A�R@������
�QG�����C�8R@����'
=����� ��C�n                                    Bxj;�&  �          A�R@����R�[�����C���@�������H�  C�4{                                    Bxj;��  T          A=q@�z���G��e���\)C�9�@�z������Q��p�C�
=                                    Bxj;�r  �          A
=@����  �_\)���
C���@�������p���
C�1�                                    Bxj<  �          A33@�Q����H�[���(�C�Ff@�Q��   �����Q�C��=                                    Bxj<�  �          A\)@����Q���ffC��@���H��ff����C�@                                     Bxj<(d  �          AQ�A��i���G
=����C��A���ff��{�ޣ�C�Ф                                    Bxj<7
  �          A  A���W
=�HQ���  C��A�ÿ��
���\����C�Ǯ                                    Bxj<E�  �          A��A33�X���<�����C���A33�������z�C�Y�                                    Bxj<TV  �          A(�@�ff���ÿ�p��1�C�'�@�ff���������C��q                                    Bxj<b�  �          Aff@u�33�\)�^�RC�5�@u��\)������C���                                    Bxj<q�  �          A�R@�G���
�k����C��
@�G����H��  ��C�Ф                                    Bxj<�H  �          Ap�@�z����k�����C�8R@�z���ff����ffC�ff                                    Bxj<��  
�          Ap�@�Q��(��fff��G�C���@�Q�������z���(�C��R                                    Bxj<��  T          A(�@�z��\)��ff��33C��q@�z�������  ��(�C�33                                    Bxj<�:  �          A  @�G��ff�u����C�=q@�G���Q���z���{C���                                    Bxj<��  �          AQ�@�G��=q�����(�C�>�@�G���p�������\C�ٚ                                    Bxj<Ɇ  �          A��@�(���׿�33�{C��)@�(�����33� �C�b�                                    Bxj<�,  �          A@�
=�z��Q��:�RC�33@�
=������\�(�C�u�                                    Bxj<��  �          A(�@���� ���p��pz�C�U�@������R��{��HC�7
                                    Bxj<�x  �          A@�������\�*=qC�\)@�����p���Q����
C�\                                    Bxj=  �          A{@����{���G\)C��
@����33��=q�=qC�޸                                    Bxj=�  �          Ap�@�=q���
=q�P  C�W
@�=q��33��p��	�RC�o\                                    Bxj=!j  �          AG�@����(��j=q���C�(�@����=q�ҏ\�-�C�q�                                    Bxj=0  �          A�@��
���H�(����G�C�R@��
��������ffC�ٚ                                    Bxj=>�  �          A�H@����\�'�����C�/\@����Q��������C��                                    Bxj=M\  �          A�@�ff��ff�\)�z�HC��=@�ff��p����H�=qC���                                    Bxj=\  �          A�@���ۅ�333���C�T{@����\)��ff�C���                                    Bxj=j�  {          A�H@�ff��{�4z����C���@�ff��=q������HC�>�                                    Bxj=yN  �          AQ�@�  �ڏ\���NffC�=q@�  ������33���C��H                                    Bxj=��  �          A=q@�����Q��A����HC�&f@����������
�=qC��
                                    Bxj=��  �          A\)@�z���\�J=q��p�C��@�z�����������C���                                    Bxj=�@  �          A�@�z���\)��(��#33C��@�z������� ��C��                                    Bxj=��  �          A\)@�33���.{��=qC�|)@�33��  ��p���\C�g�                                    Bxj=  �          A{@�\)��zῺ�H��C�~�@�\)�����G���33C��                                    Bxj=�2  �          AQ�@Ǯ��Q��33�3\)C�U�@Ǯ��\)���
��33C�U�                                    Bxj=��  �          Az�@�G���
=�����7�
C���@�G���p�������ffC���                                    Bxj=�~  �          A��@�G�����{�o
=C�^�@�G������p��	�\C�N                                    Bxj=�$  
�          A�\@�=q��=q���,z�C��f@�=q��33���R��G�C�޸                                    Bxj>�  �          A(�@��H������H�eC��@��H��������Q�C��
                                    Bxj>p  �          A(�@ڏ\�У�����h��C�&f@ڏ\����������C�!H                                    Bxj>)  �          AQ�@�  �ҏ\� ���n�RC��q@�  ���\��33��C��                                    Bxj>7�  �          Az�@�p���\)�:�H���\C���@�p��Z=q��{��C���                                    Bxj>Fb  �          A�A
=�~{����O�C���A
=���u���{C�33                                    Bxj>U  �          AG�A����H��z��3�C��A��G
=�xQ���(�C��                                    Bxj>c�  �          A(�@�G���p��+���\)C�{@�G��k���=q��z�C���                                    Bxj>rT  �          A�@����(����
�ȸRC���@���Y������)�HC���                                    Bxj>��  �          A�@�z�����G���p�C���@�z��H���Ӆ�+�C���                                    Bxj>��  �          A
=@��H��  ���R���
C�N@��H�N�R��=q�*�C���                                    Bxj>�F  �          A
=@�=q��
=�w���ffC�޸@�=q�W��ȣ��!Q�C�p�                                    Bxj>��  �          Aff@�(���Q���Q�����C�\)@�(��U�����&�
C�:�                                    Bxj>��  �          A  @������{���  C���@���hQ��Ϯ�&C�\                                    Bxj>�8  T          A{@�����%��|��C���@��qG������ {C��                                    Bxj>��  �          A=qA���ff��
=��G�C�C�A��b�\�S�
���RC�w
                                    Bxj>�  �          A��@�z�����w
=��(�C�j=@�z��W������#��C�{                                    Bxj>�*  �          A�@�p���
=�aG����C�{@�p��`����\)���C��                                    Bxj?�  �          A��@�G���{���]�C�f@�G��w
=��  �C��                                    Bxj?v  �          A  @��
��33�>{����C�Ǯ@��
�k������C��                                    Bxj?"  �          A�
@׮�����G
=��G�C�ff@׮�i�����H�ffC���                                    Bxj?0�  �          A�
@�z���33�[���G�C�Q�@�z��\�����H��
C�K�                                    Bxj??h  �          A33@ָR�����S�
���C���@ָR�[���ff�\)C�w
                                    Bxj?N  �          A33@ָR�����S�
���
C���@ָR�\(����R��C�o\                                    Bxj?\�  �          A�\@�=q��p��HQ����C�  @�=q�Z�H��  �{C��=                                    Bxj?kZ  �          A��@������L(���p�C�)@���c�
��z��ffC�                                    Bxj?z   -          AG�@�=q��
=�S�
��z�C�z�@�=q�W����Q�C�h�                                    Bxj?��            A��@������\�u����C���@����1���ff�33C��=                                    Bxj?�L  �          A�H@�=q��
=�x�����HC�J=@�=q�)����ff�ffC�c�                                    Bxj?��  �          A�H@�=q��\)�w����C�Ff@�=q�*=q���{C�XR                                    Bxj?��  �          A�\@�  ��G��qG�����C�  @�  �#33��  �C�H                                    Bxj?�>  �          A\)@�������mp���G�C�q�@�����
����\)C�0�                                    Bxj?��  �          A�
@��������j�H��=qC��@����  ��\)�G�C��\                                    Bxj?��  �          A��@�Q�����vff���C���@�Q��&ff��z��  C��3                                    Bxj?�0  T          A@ᙚ���~�R�ÅC��
@ᙚ�#�
��Q���
C��                                    Bxj?��  �          Az�@�(�������H�˙�C���@�(������{C���                                    Bxj@|  �          A��@�\����c�
��G�C��\@�\�7���
=�33C��R                                    Bxj@"  �          A��@�z������{��¸RC�B�@�z��*=q�����C�q�                                    Bxj@)�  �          AG�@�p�����vff��=qC��3@�p��(�������C��R                                    Bxj@8n  
�          A��@�����
�G
=���\C��=@���:�H����\)C�@                                     Bxj@G  z          A��@����
=�N�R��Q�C�:�@���<(���z��	�C��                                    Bxj@U�  �          A��@�33����S33��C���@�33�3�
�����	�C��=                                    Bxj@d`  �          Az�@�G����5��
=C���@�G��S�
����(�C���                                    Bxj@s  �          A(�@���ff�@  ��G�C�e@��P  ������C��3                                    Bxj@��  �          Az�@�=q��{�U��  C�7
@�=q�Dz����H��
C�C�                                    Bxj@�R  �          A�@�z���z��W�����C�xR@�z��@����33�\)C���                                    Bxj@��  �          A��@��H��  �L�����HC�q@��H�K���  �33C���                                    Bxj@��  �          A��@߮����\�����C��@߮�C33��
=�ffC�8R                                    Bxj@�D  �          A��@�{��
=�dz����
C�޸@�{�>{��=q�p�C�ff                                    Bxj@��  �          A�
@��H��=q�aG����C�k�@��H�E���=q�G�C��                                     Bxj@ِ  �          A  @ڏ\���R�Vff��
=C�@ڏ\�Q���  �C�*=                                    Bxj@�6  �          A�@���������p�����C�e@�������33�(��C��                                    Bxj@��  
�          A��@��H��Q���b�RC��@��H��Q������\)C��                                    BxjA�  �          A�@Ӆ��(��G��^{C�@Ӆ����������HC�33                                    BxjA(  �          A(�@�����
�4z���(�C���@���k����
�	�
C���                                    BxjA"�  �          AQ�@�=q��(��(��m�C�@�=q��33��p��(�C���                                    BxjA1t  �          A\)@��
��p��J�H����C�7
@��
�U����\��RC��                                    BxjA@  �          A�
@�=q�\�\)�t  C�%@�=q��Q���ff�33C��H                                    BxjAN�  �          A��@�
=��p���{���C���@�
=��G������#C��H                                    BxjA]f  �          A@��������{��RC�q@�׿^�R��  �*Q�C�s3                                    BxjAl  �          A@�  �#33���R��C�K�@�  ?(����\)�"�@�p�                                    BxjAz�  �          Aff@��
�(���z��&�C�+�@��
?����(��-�RA-�                                    BxjA�X  	�          A  @��_\)��G��{C�\@��k���G��)G�C��                                    BxjA��  �          A��@�G��tz���Q��G�C�*=@�G��(����
=�%�
C�g�                                    BxjA��  �          A�@�����
��ff��G�C�:�@�׿�����33�#ffC�Ф                                    BxjA�J  T          AQ�@������R��=q���RC��@��ÿ�ff��(��+G�C��                                     BxjA��  �          A�
@��������\���C��\@��ÿ������H�>ffC��R                                    BxjAҖ  "          A\)@����H�K����
C���@����H��=q�{C�C�                                    BxjA�<  T          A ��@Ӆ��
=�<(���Q�C���@Ӆ��������G�C�Ǯ                                    BxjA��  �          A#�@�Q���  �E���z�C���@�Q���ff��=q�C�
                                    BxjA��  T          A&{@�z������]p���{C�h�@�z���z���\)�!�RC��                                    BxjB.  �          A(z�@��
��\)�j=q����C�/\@��
���\����(��C��                                     BxjB�  �          A)�@�  �����R���RC�4{@�  ��Q���\)�-�C�'�                                    BxjB*z  T          A)�@ə����<�����C�8R@ə���33�љ����C�Ф                                    BxjB9   "          A)p�@�z��
=q���O\)C���@�z���  ��p��Q�C�                                    BxjBG�  "          A*�R@��H��Ϳ�Q��'33C�C�@��H��=q��  �Q�C�                                    BxjBVl  �          A,  @�z��\)�Q��7�C��@�z�����Q����C�>�                                    BxjBe  �          A+�
@�����
�!G��Yp�C��\@�����G���\)�C�xR                                    BxjBs�  z          A,(�@�\)���,(��h��C��)@�\)��{���
���C��q                                    BxjB�^  T          A,  @�  �  ��Q���(�C�K�@�  ������33���C�k�                                    BxjB�  |          A+�@���Ӆ?���A(��C���@�����
����&�HC���                                    BxjB��  �          A*�HA���  @fffA��C��
A���p�?0��@n�RC��=                                    BxjB�P  �          A+�
A=q��G�@�A-�C�}qA=q���Ϳ0���l��C���                                    BxjB��  �          A/33A�����\@  A>=qC�\A����(�������C�                                    BxjB˜  �          A0(�AQ����@Z�HA��\C��\AQ���  ?8Q�@o\)C�                                      BxjB�B  �          A/33A���ff@`��A��C�H�A����
?@  @|��C���                                    BxjB��  �          A.=qA33���
@L(�A�=qC��3A33��z�?�\@,��C�&f                                    BxjB��  �          A-�A�����@`  A�\)C���A����{>��@	��C���                                    BxjC4  T          A)�A\)��  @Q�A�{C�}qA\)���<��
=��
C�p�                                    BxjC�  �          A&{@�����@�G�A��C��H@����33>���?���C�XR                                    BxjC#�  �          A"�R@ٙ���Q�@Q�A�
=C���@ٙ���\)�!G��e�C�"�                                    BxjC2&  T          A Q�@�Q���33@0��A�
C�]q@�Q���\��  ��  C�Z�                                    BxjC@�  �          A z�@��ʏ\?�  A!p�C�H@����H��p���C��q                                    BxjCOr  T          A�A=q���ÿ�33�޸RC���A=q�aG��\(���p�C�P�                                    BxjC^  �          A�
@߮�	���Ǯ�!��C�xR@߮?����θR�(�HA/\)                                    BxjCl�  .          AQ�@�Q�h����  �6�C�/\@�Q�@3�
��
=�%�A�Q�                                    BxjC{d  z          AQ�@�{��p���
=�-�
C���@�{@
�H�����'��A�
=                                    BxjC�
  �          A��@ۅ��  �׮�.Q�C���@ۅ?������,z�A~ff                                    BxjC��  �          AQ�@�33�\)����!33C�:�@�33?����=q�)33A'�
                                    BxjC�V  �          A�@陚�   �ʏ\�C�Y�@陚?���θR�#��A>ff                                    BxjC��  T          A@�׿�����z��!Q�C���@��?�{��\)�$(�AH                                      BxjCĢ  �          A�@�p����H��33�((�C�AH@�p�@������"Q�A��H                                    BxjC�H  �          A\)@�=q�
=��p�� z�C���@�=q?��R���H�%��A8z�                                    BxjC��  T          A�@�=q�����(���\C��@�=q?��\��z���AQ�                                    BxjC�  �          A@��
����=q��HC���@��
?�������H@���                                    BxjC�:  �          A�@�33�+�������\C��3@�33?
=q�Å�\)@~{                                    BxjD�  �          A   A=q�!G����R���C�b�A=q?������{@j�H                                    BxjD�  �          A z�A (��(���Q��ffC��A (�?@  �����@�=q                                    BxjD+,  T          A"{@�z��9�������
C���@�z�>�(����H�
=@H��                                    BxjD9�  �          A"{@�
=�;������(�C��=@�
=>�{�Ǯ�(�@(�                                    BxjDHx  T          A#
=Az��:�H�����C�H�Az�>#�
���p�?��                                    BxjDW  �          A ��@�(�������
=��\)C��3@�(���  ���\�  C���                                    BxjDe�  �          A"{A �����
�U��(�C�˅A ���*=q��  ���C��                                     BxjDtj  �          A#33A�
��p��]p���z�C���A�
����Q����C���                                    BxjD�  �          A$��A������s�
��=qC�\)A���G���p���C�*=                                    BxjD��  �          A#33A Q����H�y������C�q�A Q��Q����H�z�C���                                    BxjD�\  z          A$z�@��\���H�i����=qC�>�@��\�8����Q��{C��)                                    BxjD�  "          A"=qA33�qG����H�ׅC���A33�Y����(���C��                                    BxjD��  T          A"�RA  ��33������\)C��fA  �����G��
�HC��=                                    BxjD�N  �          A"ffA\)�N�R������Q�C�>�A\)��Q������
�C��                                     BxjD��  T          A ��A����G�����G�C���A��?�
=���
�
=AV�R                                    BxjD�  �          A ��A�
�#33�����33C�h�A�
?   ��\)�(�@\��                                    BxjD�@  �          A"ffA\)�B�\��\)��=qC��A\)�\)���
�z�C���                                    BxjE�  "          A$��A	�dz����
�ȸRC��)A	�J=q���\�p�C�aH                                    BxjE�  |          A)G�AQ��Z=q��Q����\C���AQ�333������HC��                                    BxjE$2  �          A)��A���g
=��{��{C�qA�׿h����ff���C��                                    BxjE2�  �          A'\)A=q�B�\��=q��p�C���A=q�k������\)C�E                                    BxjEA~  �          A'�A�R�N{��z��ƸRC�{A�R�����p���{C�~�                                    BxjEP$  �          A'\)A
=����XQ����C�Q�A
=��{���H��G�C��                                    BxjE^�  �          A%��A������I����  C�8RA��  �����z�C���                                    BxjEmp  z          A$(�A33���R�G
=��33C�p�A33�(���\)��{C��                                    BxjE|  �          A%G�A�������S33��
=C��A�ÿ�Q�������C��                                    BxjE��  �          A%�A��w��l(����
C�*=A�������p���ffC��                                    BxjE�b  �          A&�RAG��y���z�H����C�RAG�������
��ffC�ٚ                                    BxjE�  
�          A&�HA
=q�\)������\)C���A
=q��
=��
=��C��                                    BxjE��  �          A'
=A��b�\��ff���
C��A녿Tz�������C�P�                                    BxjE�T  T          A!A\)�E���\�ׅC��
A\)��  �����ffC�%                                    BxjE��  	�          A�AG��QG���(���C��
AG��\�����C��                                    BxjE�  �          A  @�ff�5���Q����C��@�ff>������R��@�H                                    BxjE�F  �          A\)@�Q��I������C���@�Q����33�  C��\                                    BxjE��  T          A�@���G���ff��G�C�8R@���aG���z���
C�8R                                    BxjF�  �          A�@����Dz���Q�����C�]q@��;���p����C���                                    BxjF8  �          A��A  �Y��������=qC�˅A  �L�����R� �\C�:�                                    BxjF+�  �          AQ�A Q��`  �����  C�8RA Q�:�H��  �	p�C�g�                                    BxjF:�  �          AA��p  �|������C�A��������\��RC�
                                    BxjFI*  �          AAp��j=q�n�R��(�C�#�Ap�����33��\)C��q                                    BxjFW�  �          A�A���^{�qG���C��A�׿}p�������\C��\                                    BxjFfv  �          A33A
=q�_\)�Y����=qC���A
=q������
=��\)C��                                    BxjFu  �          A (�A��z�H�;���(�C��A���������Q�C�"�                                    BxjF��  �          A!G�A	����ff�p����=qC�h�A	���|(��aG���G�C���                                    BxjF�h  �          A ��A	����������33C���A	��r�\�e���G�C�\                                    BxjF�  �          A ��A����p���33��=qC�A���XQ��Z�H����C�~�                                    BxjF��  �          A=qAp��n{�@  ���RC��fAp�������  ����C�e                                    BxjF�Z  �          A(�@��\�QG������Q�C���@��\�����\���C��=                                    BxjF�   �          A�@����`  �������C��
@��Ϳ=p���\)�  C�@                                     BxjFۦ  �          A�\@����l������33C��3@��ͿW
=���R���C��f                                    BxjF�L  �          A�@�����(�������C��@��Ϳ�G���\)�'�C��3                                    BxjF��  �          A��@�{�C�
��Q�����C�(�@�{��G������=qC��{                                    BxjG�  �          A
=@���7
=��\)��Q�C��\@�녽�Q������C��=                                    BxjG>  �          A�H@�G��E���(���RC��f@�G��L�������ffC�AH                                    BxjG$�  �          A��@�Q��G
=������33C��)@�Q쾏\)�����C��3                                    BxjG3�  �          A{@�ff�^�R�����G�C��3@�ff�W
=�����	33C��                                    BxjGB0  �          A=q@�\)�]p�������HC��R@�\)�z���z��C���                                    BxjGP�  �          A�R@��\����ff��  C��@��:�H��z����C�E                                    BxjG_|  �          A@��qG��r�\��
=C��@���(���ff�{C�~�                                    BxjGn"  �          Az�@�ff�`����  �ڣ�C�aH@�ff�@  ���R��C��                                    BxjG|�  �          A��@�\�x���h����33C�g�@�\��33��z���HC���                                    BxjG�n  y          A��@�(�����Fff��33C���@�(��Q����R� 33C�5�                                    BxjG�  �          A�@�G���33�>�R��G�C�L�@�G��ff��ff� p�C�XR                                    BxjG��  �          A��@�����{�I����  C���@������z��{C�<)                                    BxjG�`  �          AG�@�p���G��/\)���HC���@�p���H��
=��\)C�9�                                    BxjG�  �          Ap�@�ff���\����S�
C��@�ff�?\)����ޣ�C�c�                                    BxjGԬ  �          A��@�=q������ff�1p�C�8R@�=q�H�������ͅC�\                                    BxjG�R  �          A=q@��
������L��C��@��
�7���
=��C��
                                    BxjG��  �          A�@�������33�$  C��)@����b�\��(����C�j=                                    BxjH �  �          Ap�@���=q�W
=��z�C��3@������h����C���                                    BxjHD  �          Az�@�ff��(�������C��@�ff�x���|(����C�4{                                    BxjH�  �          AQ�@�����=q��
=�陚C�O\@���������ff��p�C�K�                                    BxjH,�  �          AG�@����zῚ�H��ffC�.@����Q�������C�33                                    BxjH;6  T          A(�@�Q���G���  �p�C��\@�Q���33��ff���C��                                    BxjHI�  �          A�@���ə��У��'\)C�/\@����  ��ff���HC���                                    BxjHX�  �          A\)@�����������!��C��=@�����(����R��C�K�                                    BxjHg(  �          A�@��������
���C�y�@�����(������RC��
                                    BxjHu�  �          A�
@��R��녿�����C�  @��R���\��
=��p�C�(�                                    BxjH�t  �          A(�@�p���������{C��=@�p���G���G���  C��=                                    BxjH�  �          A�@�����Q쿓33��33C�Ff@�����z��������HC�,�                                    BxjH��  �          A�@�ff���
�xQ���ffC��R@�ff���\��{��
=C�u�                                    BxjH�f  �          A\)@�33���aG�����C�w
@�33��������\)C��                                    BxjH�  �          A�@�=q��\)�L����(�C�O\@�=q��Q���33��ffC���                                    BxjHͲ  �          A�
@�����z�O\)����C��=@���������ff��33C�ٚ                                    BxjH�X  �          A��@����z�8Q���=qC��=@����{���
��\)C�                                      BxjH��  �          A��@����׿z��k�C�5�@����(���=q��{C�1�                                    BxjH��  
�          Az�@�����׿(��w�C�@��������33��ffC�3                                    BxjIJ  �          A��@����<��
>��C�h�@����\)�~�R���HC���                                    BxjI�  "          A@�z���Q�>\)?\(�C�1�@�z��θR�u�ǅC�C�                                    BxjI%�  �          A�@���  >\@
=C�Q�@���=q�hQ����C��                                    BxjI4<  �          A{@�\)��(�?
=q@Y��C��@�\)�أ��c�
���C�'�                                    BxjIB�  /          A�H@����33?�@c33C���@����Q��aG���z�C���                                    BxjIQ�  �          A�R@��\���?p��@��
C��q@��\��\�Q���p�C�*=                                    BxjI`.  T          A{@��\����?�G�@��RC�>�@��\�����AG����HC�R                                    BxjIn�  T          A33@�����Q�?��R@�
=C���@�����  �B�\��Q�C���                                    BxjI}z  �          AQ�@�z����?��Ap�C�4{@�z����AG���(�C���                                    BxjI�   y          A�@������?�(�A�C���@�����=q�=p���Q�C��                                     BxjI��  T          A��@��\� z�?��A1C�(�@��\�����-p���\)C��\                                    BxjI�l  �          A��@�����
=?�A7
=C�q�@�������(Q�����C�˅                                    BxjI�  �          A��@�=q� ��?�
=A>�\C�
@�=q���
�'
=���C�aH                                    BxjIƸ  �          Ap�@����\)?�z�A<z�C��@�������,�����\C�Z�                                    BxjI�^  �          A=q@�(�� ��@   AD��C�E@�(���z��"�\�{�C��H                                    BxjI�  �          A@�p���@�RA\��C���@�p���p����a�C���                                    BxjI�  �          AQ�@��H��\)@,(�A�33C��@��H������
�1G�C�>�                                    BxjJP  "          Az�@�=q���\@(�AtQ�C�n@�=q��{�33�L(�C�B�                                    BxjJ�  T          Ap�@�=q��  @\)Aw�C�C�@�=q���Ϳ��H�B{C�f                                    BxjJ�  �          A��@�����@ ��AyG�C�ff@�����Ϳ�Q��?\)C�"�                                    BxjJ-B  �          AQ�@���ff@(�At  C�@����
�����8Q�C�xR                                    BxjJ;�  �          A��@�p���z�@3�
A���C���@�p���Q��  ��
C�,�                                    BxjJJ�  y          A(�@���@@  A�33C�!H@���\)��G����\C�B�                                    BxjJY4  �          A�@{����@S33A�  C��
@{��  ��(���33C��q                                    BxjJg�  T          A�R@�
=��@.�RA�Q�C�e@�
=���R��(��-�C��{                                    BxjJv�  �          AQ�@K�����@k�A��C�p�@K��p��Tz����\C�n                                    BxjJ�&  �          A�H@AG����
?�p�A>ffC��{@AG���z��"�\���C�"�                                    BxjJ��  
�          A	�@[���Q�@(�AmG�C�C�@[���G���b�HC�:�                                    BxjJ�r  �          A��@E���H@	��Aq��C�o\@E���
���d(�C�c�                                    BxjJ�  �          A33@�G���p�?�33A{C�q@�G������!����\C��                                    BxjJ��  �          @�
=@������\=���?c�
C�R@����|�������HC�9�                                    BxjJ�d  
�          @љ�@�p��S�
�k��z�C�G�@�p����!�����C�\)                                    BxjJ�
  �          @�{@�=q�hQ�L�Ϳ�33C�  @�=q�?\)�����C�g�                                    BxjJ�  �          @ᙚ@�Q���R�z�H�
�HC���@�Q쿱�� ����=qC��H                                    BxjJ�V  �          @�
=@�z�>k��'����@(�@�z�?����
=����An�H                                    BxjK�  �          @�ff@�ff>�  �����=q@33@�ff?�  ������{AYp�                                    BxjK�  y          @�@���?�p��S33���ADz�@���@1��
=q��A�                                    BxjK&H  T          @��@�  >����1G���33@Q�@�  ?�p�������A�=q                                    BxjK4�  �          @��@�Q�?(���{���
@ȣ�@�Q�?�{��G��yp�A��                                    BxjKC�  /          @ٙ�@�
=@z���R���A���@�
=@C�
���
�\)A�33                                    BxjKR:  y          @��H@��@���#�
��(�A�p�@��@W��k�����A                                    BxjK`�  T          @ٙ�@ƸR?��׿�=q����A%��@ƸR?��u�{A��                                    BxjKo�  �          @�\@�녿��R���[�C��@�녿Q��{���C�s3                                    BxjK~,  T          @�z�@�\)��\)���H��Q�C�G�@�\)=��
�����=q?(��                                    BxjK��  �          @�(�@�p��8���
=���C��@�p������S�
��Q�C��                                    BxjK�x  "          @�p�@Ӆ��
����C�^�@Ӆ�8Q��N�R��\)C���                                    BxjK�  �          @�ff@�\)>���=q����@�@�\)?�p����x��AF�H                                    BxjK��  �          @�Q�@�33���R����RC��{@�33?u�	����ffA                                       BxjK�j  �          @�p�@�>�G�������@q�@�?У׿�ff�k
=A[33                                    BxjK�  �          @�{@ָR=u����33?�@ָR?�=q��\����A3
=                                    BxjK�  T          @��@��;�{�\)��  C���@���?aG�����@�G�                                    BxjK�\  �          @���@�z�L���q���33C�@�z�?�Q��P  ��=qA�=q                                    BxjL  y          @�
=@��R���{��33C�@ @��R?����i����
=Alz�                                    BxjL�  �          @��@�=q�������C�޸@�=q@33�r�\� �\A�p�                                    BxjLN  T          @��
@��?���(����@�\)@��@'
=�P  ���A�                                    BxjL-�  �          @��@�33>aG���=q�G�@
=@�33@�
�Vff���A��R                                    BxjL<�  �          @���@�z�?#�
�tz�����@�{@�z�@"�\�:�H��p�A��
                                    