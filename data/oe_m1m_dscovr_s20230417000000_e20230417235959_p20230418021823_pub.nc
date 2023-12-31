CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230417000000_e20230417235959_p20230418021823_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-18T02:18:23.387Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-17T00:00:00.000Z   time_coverage_end         2023-04-17T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxx�.   �          @��H�\)���H?˅A;\)C}  �\)������
��
=C|��                                    Bxx�<�  
�          @�������?�  Ap�C�&f����G��(Q���\)C�)                                    Bxx�KL  �          @�33���R�陚?#�
@��C~�R���R��(��G
=��\)C}E                                    Bxx�Y�  
�          @�\��\���>�33@)��C~T{��\�θR�U��  C|}q                                    Bxx�h�  �          @�\��{���?xQ�@���C���{�����333���
C~��                                    Bxx�w>  T          @��H�{��p�?�{A�\C|�=�{��Q��(Q���Q�C{�\                                    Bxxȅ�  T          @����33�陚?z�H@�33C~L��33����3�
��z�C}:�                                    BxxȔ�  "          @���
=q���H?   @p  C}�H�
=q��33�O\)��  C{�                                    Bxxȣ0  �          @��333����?�=qA"{Cw�f�333��z��33����Cw�                                    Bxxȱ�  �          @�����=q��Q�@7�A�{Ci&f��=q��p��Ǯ�:=qCl)                                    Bxx��|  �          @�\)������33@qG�A���C]{�������
?xQ�@��Cd�                                    Bxx��"  �          A z����
��(�@~�RA��
CXO\���
��=q?��A�HC`��                                    Bxx���  �          @�(���  �7�@�=qB��CN�=��  ����@�
A��RCZ�                                    Bxx��n  �          @�ff��{��ff@tz�A�  C[���{����?��HAz�Cb�                                    Bxx��  
�          @��H��ff��33@]p�A�G�C^�f��ff���R?5@�33Ce�                                    Bxx�	�  T          @�������@�\A}��Ch{���������ffCiT{                                    Bxx�`  �          @�R�|(���?�33A-p�Cmn�|(�������{�g\)Cl��                                    Bxx�'  �          @�������ff?��A�Cm�������p��	�����Ck�{                                    Bxx�5�  "          @�  �fff��z�?���A#\)Cp���fff��ff��\�}�Co��                                    Bxx�DR  �          @����������?�Q�A0z�Cj��������ff��  �W\)CjJ=                                    Bxx�R�  "          @�z���G���@7�A�G�C`Y���G���ff=#�
>�z�Cd��                                    Bxx�a�  "          @��H��G���p�@J�HA�(�Cd� ��G���=q>8Q�?�\)Ci5�                                    Bxx�pD  T          @���G����\@J�HA��HCg33��G���ff=���?B�\CkW
                                    Bxx�~�  T          @�p��������@�
A�ffCi�q��������L���ȣ�Ck�
                                    Bxxɍ�  �          @�\)������G�@G
=Aď\Cg\������z�=�\)?��Ck#�                                    Bxxɜ6  �          @��������\@;�A�p�Cfff����\��G��O\)Cj�                                    Bxxɪ�  �          @�����\���@z=qA���C`h����\���H?�A�RCg�)                                    Bxxɹ�  T          @�\)�������@vffA�ffC^)�����(�?�(�A=qCe�f                                    Bxx��(  
�          @��������r�\@���B	
=C\�������G�?�{AP��CexR                                    Bxx���  �          @�
=��(��r�\@dz�A��\C[Q���(����?�(�A!Cc�\                                    Bxx��t  "          @��H�������@+�A�ffC[�)������{>�=q@
=qC`}q                                    Bxx��  
�          @�{��G�����@)��A���Ca
��G�����=L��>�ffCeE                                    Bxx��  
�          @أ����\���?�
=Af�HCgc����\���׿����ChE                                    Bxx�f  
�          @���������?�z�A�\)Cb����������5���Cds3                                    Bxx�   T          @����������?�z�A|��Cb� ������ff�B�\��
=Cd8R                                    Bxx�.�  T          @�R��G��g�@_\)A�RCW����G���(�?�  A!G�C`                                    Bxx�=X  
�          @�
=��{�0��@��B�CQ+���{���@#33A�\)C_�                                    Bxx�K�            @�=q�����`  @�G�B\)CX��������@ ��A~{Cc
                                    Bxx�Z�  
�          @�  ���\�b�\@��
B
�CX8R���\���?�Aj�RCb�)                                    Bxx�iJ  
�          @ۅ���H�k�@L��A�  CYJ=���H����?xQ�AC`��                                    Bxx�w�  
�          @أ��������@�  B  CC������N�R@1G�AîCT\                                    Bxxʆ�  �          @�����:�H@�B�C<{����,(�@Q�A�
=CO��                                    Bxxʕ<  
�          @�(���\)�aG�@z�HB
=C=���\)�,(�@?\)A�\)CO@                                     Bxxʣ�  T          @�z���
=�s33@z�HB�
C>Y���
=�0��@<(�A�(�CO�
                                    Bxxʲ�  
Z          @��
��33�5@�Q�B33C<^���33�%@I��A�{CP�                                    Bxx��.  �          @�(�����?#�
@�\)B+�HC,L����׿��@�
=BCF�R                                    Bxx���  
�          @�
=�Tz�?�33@��
BZ�C=q�Tz�z�H@��
BiG�CDk�                                    Bxx��z  �          @����j=q?��@��B[�\C#�H�j=q��p�@��BQ=qCM^�                                    Bxx��   �          @��
�w
=?��
@��RBR
=C%
=�w
=��33@���BH��CK.                                    Bxx���  T          @���R?���@�\)B��\C@ ��R���@���Bw
=CY@                                     Bxx�
l  T          @�ff��33?�
=@�  B��C����33�˅@ȣ�B���C_�                                    Bxx�  �          @�\)�'�?���@��HB��fCE�'��G�@�33Bq�CY��                                    Bxx�'�  T          @�\)�N{?��@���Bnz�C!Y��N{���@��HBa{CRY�                                    Bxx�6^  �          @����
=?0��@�=qB@��C+8R��
=��\)@�Q�B1�CJ�R                                    Bxx�E  T          @�\)����?�\@�B<�C-�\�������H@���B*ffCKJ=                                    Bxx�S�  �          @�  ��  >�(�@���B?�
C.����  �z�@��HB+33CL��                                    Bxx�bP  �          @ҏ\�(����H@��B��CN\)�(��~{@�z�B3ffCns3                                    Bxx�p�  �          @�(��9������@�G�BrCIO\�9���l(�@�G�B)Cg��                                    Bxx��  �          @ȣ��q녾��@��BQ�C:#��q��0  @��HB$�CX�                                    BxxˎB  �          @�\)�xQ��ff@��HBL\)C:���xQ��/\)@�Q�B ��CWJ=                                    Bxx˜�  T          @�p��O\)��
=@�BU�
COT{�O\)�w
=@_\)B
��Ce��                                    Bxx˫�  T          @ƸR�E��
=q@�p�BRQ�CW\�E�����@P  A��Cj@                                     Bxx˺4  �          @��
�(���<(�@�ffBM�Cd{�(����\)@9��A��
Cr�                                    Bxx���  T          @�\)�Mp���
=@�  BY�HCS�Mp���\)@i��B
  Ch��                                    Bxx�׀  �          @�=q�K�����@�G�Bj�
CG
�K��i��@��B&Cd�f                                    Bxx��&  �          @У��<�Ϳ�@�Q�Bf�CS��<������@{�BQ�Ckff                                    Bxx���  �          @����@����@���B_�CV���@����@mp�B  Ck�
                                    Bxx�r  �          @У��>{�	��@��
B^�
CW���>{��\)@j=qB	{Cl��                                    Bxx�  �          @��H�>{���@�p�B]��CYO\�>{��33@i��B�Cm#�                                    Bxx� �  �          @Ӆ�C33�@�(�BZ  CY� �C33����@e�BQ�Cl�                                     Bxx�/d  �          @�z��<(����@��
BT�
C[�
�<(����
@S33A���Cm��                                    Bxx�>
  �          @�z��2�\��@�Q�B^z�C[E�2�\����@`��B��Cnk�                                    Bxx�L�  �          @�
=�,(���\@�
=Bdp�CY!H�,(�����@eB33Cm��                                    Bxx�[V  �          @������?��@�Q�B'��C%�{��녿xQ�@�G�B)�C@�R                                    Bxx�i�  �          @�  ����@�@dz�BG�C������>���@�  B#p�C00�                                    Bxx�x�  �          @ҏ\���@#�
@eB��C@ ���?�@��
B'Q�C-�                                    BxẋH  �          @Ӆ��{@:�H@O\)A��Cp���{?�  @�  B z�C(�\                                    Bxx̕�  T          @�
=��p�?�(�@{�B&�C ����p����@���B2G�C<33                                    Bxx̤�  �          @�z����@)��@G
=B �
C�����?Tz�@�  B-
=C(�R                                    Bxx̳:  "          @����33@Z�H@"�\A�p�C0���33?��
@w
=B#\)C�                                     Bxx���  �          @��\�u�@<��@O\)B�Cff�u�?��@�Q�B;�HC$��                                    Bxx�І  �          @�(�����@>{@`  Bp�C�R����?p��@��B<��C&�q                                    Bxx��,  �          @�z���Q�@Z�H@HQ�A��C����Q�?�  @�(�B6�\CxR                                    Bxx���  �          @���o\)@e@333A�33C+��o\)?���@�B4�\C�                                    Bxx��x  �          @�\)�X��@g
=@7
=A�(�CE�X��?�ff@�\)B?�C�                                    Bxx�  �          @�����H@Dz�@1�A�ffC����H?�\)@z=qB(C!u�                                    Bxx��  �          @�ff�z�H@c�
@,��A�=qC	� �z�H?�@�=qB,��C��                                    Bxx�(j  T          @�
=�mp�@}p�@��A�Q�C��mp�@z�@�G�B*�C�                                    Bxx�7  �          @�G�����@u�@ffA�G�Cz�����@  @x��B p�C�
                                    Bxx�E�  �          @������@�G�@z�A��C�
����@$z�@o\)B�\Cff                                    Bxx�T\  �          @�Q�����@�G�?�=qA��C�����@,(�@a�B  Cn                                    Bxx�c  �          @�\)�e@���?��AQC }q�e@U@U�B��C	�                                    Bxx�q�  �          @��R�w
=@�p�?Tz�Ap�C0��w
=@_\)@5�A�  C	��                                    Bxx̀N  �          @�\)�u�@���?.{@��HCE�u�@j=q@0  A�33CQ�                                    Bxx͎�  �          @���L��@��R?+�@�p�B�L��L��@u@4z�A�C�\                                    Bxx͝�  �          @�p��N�R@��?�@��\B��3�N�R@���@5�A��HC �\                                    Bxxͬ@  �          @�  �J=q@��?k�A�B����J=q@}p�@J�HB �C ��                                    Bxxͺ�  �          @����>{@���?�G�AI�B�u��>{@h��@XQ�B{C:�                                    Bxx�Ɍ  �          @��
�B�\@�{?h��A
=B�G��B�\@z�H@HQ�B
=B��{                                    Bxx��2  �          @��@  @�\)?fffA��B�\�@  @~{@I��BQ�B�#�                                    Bxx���  �          @�Q��[�@��H@ffA�ffC O\�[�@.{@��
B-{C��                                    Bxx��~  T          @�p��XQ�@���@��A��
B�u��XQ�@7�@�  B/
=C�                                    Bxx�$  �          @�
=�`  @���@,��A�\)C��`  @G�@���B6�
C�3                                    Bxx��  �          @���R�\@c33@%�A�=qCٚ�R�\?�@|(�B7�
C��                                    Bxx�!p  
�          @����O\)@y��?�A��HC�3�O\)@&ff@[�B�CG�                                    Bxx�0  �          @�  �Dz�@��\?�33Alz�B��Dz�@J�H@QG�B
=C&f                                    Bxx�>�  T          @�z��P  @�33?޸RA�33B��\�P  @P  @l(�B  C
=                                    Bxx�Mb  T          @���>{@�\)@ ��A�ffB�{�>{@1�@��B:�RC�
                                    Bxx�\  "          @��\�z=q@w
=?��A��RC\)�z=q@$z�@X��B�C��                                    Bxx�j�  �          @��
�u@xQ�@��A�G�C��u@�@l��B��C��                                    Bxx�yT  �          @��R�o\)@���@  A�C��o\)@!�@xQ�B"��C�)                                    Bxx·�  T          @�(��y��@+�@fffB�Cu��y��?0��@�{BA
=C)�f                                    BxxΖ�  �          @�(���z�?���@�\)B2�
CO\��z��@��BDG�C:�\                                    BxxΥF  �          @Ǯ��\)?�z�@~{B�C����\)�\@�G�B.�\C8��                                    Bxxγ�  T          @����
?���@�=qB$�\C�f���
���@�33B2��C:0�                                    Bxx�  �          @�(����?5@��B2z�C+��������@���B)p�CE�3                                    Bxx��8  T          @˅��?��\@�B1=qC'{����z�@���B/z�CB�f                                    Bxx���  T          @��H����?5@��
B;ffC*}q���׿�G�@�p�B1\)CGp�                                    Bxx��  T          @���N�R�   @��B`=qC<�{�N�R�,��@��\B0Q�C[��                                    Bxx��*  T          @\�w�����@�{BI��C5n�w����@��
B*\)CR@                                     Bxx��  �          @Å���\>W
=@���B?�
C1\���\��z�@��
B)��CM\                                    Bxx�v  �          @�Q��q�<�@�p�BL=qC3�{�q��Q�@��B/33CQu�                                    Bxx�)  "          @��}p�>���@�(�B?ffC/�f�}p���  @���B+CK��                                    Bxx�7�  �          @�z��tz�>�z�@�(�B<��C/���tz��33@r�\B)�CKY�                                    Bxx�Fh  
Z          @�z��fff�!G�@���BU�C=���fff�4z�@��B&p�CZ
=                                    Bxx�U  �          @�(��`�׿Y��@�p�BV�HCA���`���AG�@~�RB"��C\��                                    Bxx�c�  T          @����N�R�L��@�(�Bd�CA�H�N�R�C�
@�{B-
=C_�                                     Bxx�rZ  T          @��
�_\)�.{@�\)BN33C>���_\)�)��@j�HB��CYE                                    Bxxρ   �          @�����?Y��@|��B.ffC(:���녿��@y��B+�\CB��                                    BxxϏ�  "          @�����
=?��@x��B%p�C"����
=��R@��B.\)C<J=                                    BxxϞL  "          @��\��Q�?��@n�RB  C!�f��Q���@|��B*�C:@                                     BxxϬ�  
Z          @������H?z�@�p�B5(�C+�q���H���@~{B*p�CF��                                    Bxxϻ�  
�          @��R�u��Ǯ@�BD(�C9޸�u���@p��B�\CS��                                    Bxx��>  �          @���z=q��R@��B;��C<���z=q��R@_\)B
=CTc�                                    Bxx���  T          @�=q�vff�z�@��B?��C<���vff��R@dz�B�RCT��                                    Bxx��  "          @�Q��s33���@��BAz�C;
�s33�Q�@g�B�CT�                                    Bxx��0  �          @��R�c33��R@�p�BK\)C=�3�c33�#�
@j=qBz�CW�)                                    Bxx��  "          @�ff�dz�#�
@�z�BI��C>.�dz��$z�@g�B��CW��                                    Bxx�|  T          @�{�\(��Tz�@�ffBM�CA� �\(��0  @eB��CZ�f                                    Bxx�""  �          @�\)�P�׿�{@��HBTp�CF޸�P���Dz�@eBC_O\                                    Bxx�0�  T          @�(��K���=q@���BUCF�{�K��@��@c33B  C_}q                                    Bxx�?n  "          @�z��L�Ϳ��@�Q�BT�CF���L���@��@a�BG�C_8R                                    Bxx�N  
�          @�z��H�ÿu@��BY{CD�3�H���:=q@h��B!ffC^�)                                    Bxx�\�  "          @�������@��Bb�RCWJ=���^�R@K�B=qCk�\                                    Bxx�k`  "          @�  ��׿���@�  Blz�CT�)����Tz�@W
=B��Ck��                                    Bxx�z  "          @�Q��
�H�Ǯ@�  Bm=qCWǮ�
�H�Z�H@Tz�BG�Cm��                                    BxxЈ�  �          @��׾�\)�$z�@p��B]��C��쾏\)�~�R@��A��C�f                                    BxxЗR  
�          @��
@
�H�L(�@:=qB\)C��@
�H���?�A`��C��                                     BxxХ�  T          @�ff��<#�
@L(�B	G�C3�H����@7
=A�z�CD�                                    Bxxд�  �          @�33��=q>��@O\)B\)C1�R��=q���@=p�B�CD��                                    Bxx��D  
�          @�
=�Dz���@w
=B?�\CP#��Dz��L(�@-p�A�Cb)                                    Bxx���  T          @�z��mp��u@qG�B2{CB� �mp��#33@<��B��CVk�                                    Bxx���  �          @��
�|(��W
=@]p�B"��C@��|(���\@.{A��RCR5�                                    Bxx��6  �          @��
��G�?:�H@^{B �C)�R��G��c�
@[�B�
C@xR                                    Bxx���  T          @��R���;��
@fffB�C88R���Ϳ��@Dz�BCKB�                                    Bxx��  T          @�p���
=��@p��B%��C;����
=��@G
=B�HCOG�                                    Bxx�(  �          @�33�c�
��G�@~{B@p�C5�{�c�
��{@`  B$p�CO��                                    Bxx�)�  �          @����QG��E�@���BH�RCA+��QG��{@QG�BQ�CY�                                    Bxx�8t  "          @��hQ�>��@�Q�B?  C/��hQ��ff@mp�B,�
CK�                                    Bxx�G  
�          @�=q�b�\?xQ�@r�\B7z�C$�R�b�\�O\)@u�B9�
C@ٚ                                    Bxx�U�  �          @�  ���\?�@:=qB�
C-\���\�O\)@5B\)C>�f                                    Bxx�df  �          @�=q��p�?�  @1G�A��C'G���p�����@:�HB�RC8@                                     Bxx�s  
�          @��R��33?�{@(Q�A��C%����33�.{@6ffB�C6.                                    Bxxс�  T          @�  ��(�?   @O\)Bz�C-n��(��z�H@HQ�B�
C@�{                                    BxxѐX  �          @�33���>�@S33BG�C.��������@J�HB
=C?��                                    Bxxў�  
�          @ȣ����ÿ(��@���B&
=C<O\������@VffB�HCO��                                    Bxxѭ�  T          @��H��{��G�@��B=�HCG� ��{�_\)@l��B�RC[��                                    BxxѼJ  
�          @�(����H����@���BE�CF�����H�\��@z=qB��C\+�                                    Bxx���  �          @�G���(����@�z�BA��CE�q��(��U@s�
B�\C[�                                    Bxx�ٖ  �          @У����
��ff@�z�BB33CE�����
�S�
@u�B�RCZ�                                    Bxx��<  �          @У����ÿ�ff@�ffBE�RCE�������U�@x��Bp�C[�                                    Bxx���  �          @�
=���H��@�(�B:�C:����H�#�
@�B{CQ5�                                    Bxx��  �          @��
�������@��
BJ��CC�����Mp�@�(�B��CZ^�                                    Bxx�.  T          @�{�u��
=@�33BTp�CE#��u�X��@���B {C]z�                                    Bxx�"�  T          @�G��z=q���H@��HBP(�CHn�z=q�hQ�@�B\)C^�                                    Bxx�1z  �          @�{��G���=q@�{BJ�CF.��G��\��@�33B��C\z�                                    Bxx�@   
�          @׮�����@�(�BF�
CG�����`  @�Q�BQ�C\s3                                    Bxx�N�  �          @����|�Ϳ�{@��\BOp�CG��|���a�@��RB�
C]�=                                    Bxx�]l  
�          @�����׿��H@���BLp�CG�������g
=@���B(�C]��                                    Bxx�l  �          @�����  ��Q�@���BGz�CM����  ����@z�HB
z�Ca33                                    Bxx�z�  �          @��H��\)��z�@��BD�RCB����\)�W
=@��\B�CX�)                                    Bxx҉^  �          @��
��
=�Tz�@��RBH�
C>z���
=�Fff@�=qB 33CV�R                                    BxxҘ  �          @�=q��{�s33@���BGC@!H��{�K�@��RB
=CW�f                                    BxxҦ�  �          @�p���\)�W
=@�=qB?�\C>���\)�B�\@�{B(�CT��                                    BxxҵP  �          @����=q��=q@�z�BC�RCAE��=q�Q�@��B33CW��                                    Bxx���  �          @�z���  ���@���BEQ�CB+���  �U@�z�B\)CX�
                                    Bxx�Ҝ  f          @�\)���Ϳu@�z�BM�C@W
�����Q�@�{B!�CX�R                                    Bxx��B  T          @�{���R���
@�G�BI�RCA����R�S33@��B�HCX�                                     Bxx���  �          @�(���ff���@���BK�RC:
=��ff�1�@�G�B)z�CS�R                                    Bxx���  "          @��H��
=��@��BKG�C5}q��
=�(�@�p�B/�CP��                                    Bxx�4  T          @�  ��=q��@�\)BQffCD���=q�X��@�
=B ��C[�\                                    Bxx��  "          @�(�����k�@�  B;(�C?Y�����=p�@�(�B33CT�                                     Bxx�*�  �          @أ����
�k�@��B7�\C?8R���
�8��@�Q�BffCT
=                                    Bxx�9&  �          @�\)���H�h��@���B*�\C>�����H�0  @n{B
=CQ�
                                    Bxx�G�  �          @�
=��(��k�@�Q�B4�
C?8R��(��6ff@z�HB33CS�\                                    Bxx�Vr  �          @�\���0��@��BJ��C<�
���<(�@�B%ffCU�
                                    Bxx�e  �          @�G���Q�aG�@�G�BD(�C?\��Q��A�@�B��CU�                                    Bxx�s�  �          @׮��{�O\)@�Q�B?�
C>Q���{�5@�ffB(�CT�H                                    Bxxӂd  �          @����Q�Tz�@���B>��C>^���Q��7
=@��RB{CTn                                    Bxxӑ
  �          @�{���\�
=q@��B5G�C:W
���\�#33@�\)B�COٚ                                    Bxxӟ�  
Z          @��
��Q�\@���B3C8\)��Q����@�z�B�
CN�                                    BxxӮV  
�          @�  �����!G�@�  B \)C:���������@tz�B�CL��                                    BxxӼ�  
Z          @�p���G�����@�B��C5���G���Q�@~�RB
=CH(�                                    Bxx�ˢ  
�          @�ff��=q?=p�@�(�B433C+:���=q��\)@�  B-�
CC��                                    Bxx��H  t          @�ff����?�p�@���BO  C#����ÿ��R@���BN�CD!H                                    Bxx���  4          @�z���{?���@�
=BH=qC&)��{���
@�BFG�CD\                                    Bxx���  T          @��H����?���@�G�B5�RC$@ ���ÿ^�R@���B:�\C>T{                                    Bxx�:  
�          @�  ����@ ��@�(�Bp�CL�����<#�
@�=qB.��C3�f                                    Bxx��  
�          @������?�@�{B#�C����ý�Q�@��HB5z�C5)                                    Bxx�#�  �          @������@
=q@�33B�
C�H����>.{@��B5�C2                                    Bxx�2,  
�          @�����
?���@�G�B+Q�C xR���
��(�@��HB8Q�C8��                                    Bxx�@�  
Z          @�
=���@�
@�=qB(ffC\)���=�G�@��HB@  C2��                                    Bxx�Ox  �          @�Q���p�@@�\)B�C���p�=u@�ffB4�C3O\                                    Bxx�^  �          @�������8Q�@�p�B@�C65������\@�z�B'��CNh�                                    Bxx�l�  �          @����(����
@��RBA=qC7���(���@��
B%��CO��                                    Bxx�{j  "          @������@�\@���B+�C޸���׾W
=@�B<�
C6W
                                    BxxԊ  T          @�  ���\@=q@���B$p�C�����\>B�\@��RB<G�C1�H                                    BxxԘ�  �          @�
=��z�?�(�@��B\)C����z�\)@�(�B.p�C5��                                    Bxxԧ\  �          @�\)���\@>{@~{B \)C  ���\?��@��\B!ffC(�R                                    BxxԶ  
�          @�  ��p�@�@�ffB=qC���p�=�\)@�B.C3@                                     Bxx�Ĩ  T          @��H����@{@��HB��C�)����>Ǯ@�ffB,�HC/��                                    Bxx��N  
�          @�G���33@\)@���Bz�C���33?#�
@�ffB\)C-��                                    Bxx���  T          @����(�@%@y��A��C5���(�?G�@��
B�C,h�                                    Bxx��  T          @�(���=q@(��@R�\Aә�C}q��=q?�=q@��\B  C)�                                    Bxx��@  �          @߮�P��?p��@�33Bs�RC#�R�P�׿�\)@��RBj\)CN^�                                    Bxx��  T          @���Z�H>��R@��Bp�C.Ǯ�Z�H���@��BY
=CS�q                                    Bxx��  �          @�G��e>�p�@�\)Bg(�C.{�e��(�@�z�BS{CP�=                                    Bxx�+2  �          @߮����>#�
@�{BZ�RC1�=�����	��@���BDQ�CP�                                    Bxx�9�  �          @�(���Q�?�p�@�ffB)p�C&+���Q�E�@���B-�
C<��                                    Bxx�H~  �          @�����R?fff@���B?z�C)+����R���@��RB;p�CC�                                     Bxx�W$  �          @�(����R>L��@��RBVG�C1L����R�
=@��BA�RCN��                                    Bxx�e�  �          @����ff>���@���B5p�C/h���ff��33@���B)��CFs3                                    Bxx�tp  �          @��
���R?��
@��A��RC"�����R?+�@:=qA���C-E                                    BxxՃ  �          @�  ��{=�G�@=p�A�(�C2���{���@0��A��
C>�\                                    BxxՑ�  �          @�z���=q���@?\)A�\CL\��=q�QG�?�A�CT�\                                    Bxxՠb  T          @���8Q����@z�A��Co���8Q���>.{?У�Cr�                                    Bxxկ  T          @��H�����H@�RA���C{@ �����ý��
�:�HC|�)                                    Bxxս�  �          @��׿����=q�����B�\Czc׿����Q�������
CxL�                                    Bxx��T  �          @�p����������¸RC{n����dz����H�8z�Cu@                                     Bxx���  T          @�  ��G���33?   @�33C�ٚ��G���=q��{��33C��q                                    Bxx��  T          @�G���p���?k�A��C�7
��p���녿��R�f=qC�,�                                    Bxx��F  �          @�\)�u��\)?���ALQ�C�B��u���ÿ�ff�"�\C�N                                    Bxx��  T          @��Ϳ�������>#�
?���C��
�������
�����C�1�                                    Bxx��  �          @����k���  ��\)�8Q�C��
�k���Q������HC���                                    Bxx�$8  T          @�\)�^�R��=q��33��{C���^�R��\)�l���(�C���                                    Bxx�2�  T          @�(��O\)�������S33C����O\)���\���
=C��                                    Bxx�A�  �          @�=q�(����H�����aG�C��H�(���z��U��C�\                                    Bxx�P*  
�          @�
=�����33������{C��f����\)�j�H�)\)C�
=                                    Bxx�^�  �          @��H>���=q�qG��+�C��>������p�C��                                    Bxx�mv  �          @�33�:=q��
=���
�N{Co���:=q��ff��R��Q�Cl�
                                    Bxx�|  �          @���(���p�?���A0(�Ce� ��(����ÿL����(�CfQ�                                    Bxx֊�  �          @�(�������R?��A  Cg������\)�����	�Cg.                                    Bxx֙h  �          @���������
?��A-��Cf#������
=�Y�����Cf��                                    Bxx֨  t          @߮���R���@   A���C^#����R���=���?\(�C`��                                    Bxxֶ�  �          @ָR�n{��G�=�G�?z�HCj���n{��p���Q����Ch�                                    Bxx��Z  �          @�ff��\)��p�@(�A�C`�H��\)��p�>��
@0  Cc�R                                    Bxx��   
�          @��H��p����@W�A��CC޸��p��.{@!G�A�ffCN��                                    Bxx��  
�          @ۅ�ə�?8Q�@#33A�\)C-���ə����
@(Q�A��C6��                                    Bxx��L  �          @أ��ҏ\=L��?���A9p�C3�
�ҏ\���?��\A-��C8!H                                    Bxx���  �          @�G��أ׾���=��
?.{C6Ǯ�أ׾��ýL�;�G�C6�{                                    Bxx��  
Z          @�33���z�H?��A��C<aH�����?+�@���C?aH                                    Bxx�>  �          @����p��^�R@�\A��C;����p���=q?�ffAS�CAٚ                                    Bxx�+�  T          @�  ��(���\)@-p�A�
=C4�)��(����@�RA�  C>!H                                    Bxx�:�  �          @��
��Q���R@Z�HA�\)CJ��Q��W
=@z�A��
CSc�                                    Bxx�I0  �          @�G���녿B�\@g
=B�HC<&f�����@C33A�p�CIY�                                    Bxx�W�  T          @��
��  ?\(�@���B�C+(���  �:�H@���Bz�C;�)                                    Bxx�f|  �          @�33���
�8Q�@��B.z�C68R���
��=q@�Q�B��CI�f                                    Bxx�u"  �          @�Q��|(���G�@��B]p�C:Y��|(��%�@���B>=qCU(�                                    Bxx׃�  �          @ۅ��  ��Q�@�ffB/=qCB���  �8Q�@z=qB\)CS5�                                    Bxxגn  �          @����{��@��B"�C9�=��{��@}p�BffCJ�                                     Bxxס  T          @���(��L��@��B%(�C<�)��(����@z�HB
p�CM��                                    Bxxׯ�  �          @������@�
@��B2��C@ ����>���@�p�BKz�C0�                                    Bxx׾`  T          @�����G�?�\)@��B9�C T{��G��Ǯ@�G�BE=qC8�                                    Bxx��  �          @߮���R@�R@�z�B�RC� ���R>�(�@��B2p�C/\                                    Bxx�۬  �          @�  ��{@�@��B��CB���{?z�@��B'�C-�
                                    Bxx��R  T          @�\��(�@AG�@s�
B�CxR��(�?���@�p�B%G�C$�f                                    Bxx���  �          @��H��{@P  @_\)A홚C�R��{?�(�@�ffB��C!��                                    Bxx��  
�          @��
����@��@ffA�{C�����@HQ�@q�B Q�C�q                                    Bxx�D  T          @�����ff@i��?��HA�(�C�
��ff@0��@<(�A�\C(�                                    Bxx�$�  �          @�(��|(�@��\?�33A8  C��|(�@W
=@%�A��C�                                    Bxx�3�  
�          @������
@p��@33A��C	�)���
@(��@a�B(�CQ�                                    Bxx�B6  
�          @����8��@�{�W
=�p�B�.�8��@��R?E�A��B�                                      Bxx�P�  �          @�p��@  @�=q�}p��0(�B�Ǯ�@  @���?z�@�  B�Ǯ                                    Bxx�_�  �          @�  �$z�@�Q쿹����G�B�8R�$z�@�Q�>�?�(�B��                                    Bxx�n(  �          @�
=��p�@����/\)��B�k���p�@�ff�J=q�   B�ff                                    Bxx�|�  �          @�G���(�@p  �	����z�B�\)��(�@��׿!G���G�B��H                                    Bxx؋t  T          @���O\)@�ff@s33B��B�=q�O\)@!G�@��\BLG�C)                                    Bxxؚ  T          @���7
=@�Q�@`��A뙚B��
�7
=@vff@���B>�
B�(�                                    Bxxب�  �          @�Q��K�@��\@O\)A܏\B�R�K�@q�@��RB4Q�C�                                    Bxxطf  �          @�\�P  @�33@7�A���B�B��P  @�@��RB&��B�Ǯ                                    Bxx��  �          @����N{@��
@�RA���B�k��N{@�=q@�ffB
=B�B�                                    Bxx�Բ  "          @��8Q�@ƸR@	��A�B�Ǯ�8Q�@�Q�@�Q�B�\B���                                    Bxx��X  �          @�33�N�R@�@�A�33B�(��N�R@��@�p�BG�B��\                                    Bxx���  
�          @���\��@�{?ǮAL��B�\)�\��@�\)@fffA��B��=                                    Bxx� �  T          @��
�r�\@���@<(�A�Q�B�
=�r�\@hQ�@���B#��C@                                     Bxx�J  
Z          @�p��k�@]p�@���B3��C�q�k�?�@��RBb
=C�R                                    Bxx��  �          @��q�@<(�@��HB@\)C
�q�?Tz�@�G�Be\)C'��                                    Bxx�,�  "          @��
�a�@,(�@�=qBM�RC��a�?�@�p�Bo�RC+�f                                    Bxx�;<  T          @�=q��{@��@l(�A�=qC0���{@-p�@��B3��C#�                                    Bxx�I�  �          @޸R�z=q@��\@4z�A�\)C
=�z=q@`  @��BC
.                                    Bxx�X�  �          @�33�{�@���?��Az�B�  �{�@�p�@:�HA�ffC�                                    Bxx�g.  �          @����Y��@��@H��A���B�Q��Y��@j=q@�Q�B.\)C�                                    Bxx�u�  �          @�z��j�H@���@i��B 
=C �)�j�H@=p�@���B;�C)                                    Bxxلz  "          @���G�@�z�#�
�uBѨ���G�@ƸR@A�BӞ�                                    Bxxٓ   
�          @��H��33@ڏ\�����Q�B�8R��33@�=q?��A(Q�B�G�                                    Bxx١�  T          @�zῥ�@�=q�����P  B�uÿ��@�ff?h��@�=qB�{                                    Bxxٰl  T          @��
�Ǯ@����>{���HB�녾Ǯ@�G�������B�L�                                    Bxxٿ  
�          @�녾�33@��H�7
=��z�B�33��33@Ǯ�5��  B�u�                                    Bxx�͸  �          @љ���Q�@�
=�E����B�8R��Q�@�{�c�
���\B�k�                                    Bxx��^  
�          @�R�E�@љ��7���z�B�z�E�@�z�����K�B�aH                                    Bxx��  
�          @�\�5@љ��N{��Q�B�W
�5@�Q�8Q���B�#�                                    Bxx���  	�          @�G��
=@У��L����ffB�Q�
=@�\)�8Q���ffB�Q�                                    Bxx�P  T          @�{��G�@�p��P����p�B���G�@��ͿQ��љ�B��q                                    Bxx��  �          @�33����@��
�!G���ffB��f����@�\�u���HB�u�                                    Bxx�%�  T          @��;L��@��
�*�H����B�aH�L��@�z�W
=��33B�#�                                    Bxx�4B  T          @�{��\)@�=q�6ff��  B�\��\)@�p�����  B���                                    Bxx�B�  �          @�\)���R@ƸR�C�
���
B�Ǯ���R@�(��:�H����B�8R                                    Bxx�Q�  "          @�?��@��
�%���RB��?��@��H�u��B��                                     Bxx�`4  �          @�?333@�{�!����B�k�?333@�z�<��
>#�
B��                                    Bxx�n�  �          @أ�?G�@�  ��  �o�B�aH?G�@ָR?
=@��B���                                    Bxx�}�  T          @��?u@�(���\)�33B���?u@Ӆ?�p�A&�\B��=                                    Bxxڌ&  T          @��H?�@��H�ٙ��n�HB�p�?�@�G�?z�@���B��q                                    Bxxښ�  �          @ٙ�?�@�33�\�O\)B�?�@�
=?Q�@޸RB��                                    Bxxکr  �          @�
=?}p�@Ϯ��ff�2�HB��3?}p�@љ�?}p�A��B��
                                    Bxxڸ  "          @�Q�?�Q�@�Q��p���33B��\?�Q�@�z�=L��>�(�B�                                    Bxx�ƾ  �          @��
?��R@ƸR�����Q�B���?��R@��H=#�
>��
B�u�                                    Bxx��d  �          @�Q�@��@�ff�Dz��ȣ�B��@��@�(��J=q��
=B�k�                                    Bxx��
  "          @���@��@�G��I���ʣ�B�aH@��@߮�W
=��G�B���                                    Bxx��  �          @��H@G�@����c�
��RB�.@G�@���p��{B��\                                    Bxx�V  T          @�\@ ��@�(��Fff���B�B�@ ��@�G��0����{B�#�                                    Bxx��  T          @�{?�  @�Q��&ff��33B�  ?�  @��.{���B��)                                    Bxx��  "          @��?Y��@�\)�0  ��33B�k�?Y��@�G��   ��{B��3                                    Bxx�-H  
�          @��Ϳ�Q�@��\>��@�Q�B�𤿸Q�@k�?�A�
=B޽q                                    Bxx�;�  
�          @�\)��{?У�@QG�B�C ٚ��{>�G�@hQ�BffC.�H                                    Bxx�J�  �          @�G�����@Q�@eBffC����?O\)@�33BQ�C*ٚ                                    Bxx�Y:  
�          @����33?L��@�(�BC,�=��33�5@���B\)C:�)                                    Bxx�g�  �          @�G���p�?k�@�  B	G�C+�\��p����@�=qB�\C9�                                    Bxx�v�  �          @�
=��p�?�ff@z�HB��C'����p���\)@�(�B�RC4�3                                    Bxxۅ,  �          @��\?��\@mp�A���C(=q�\�#�
@z�HBG�C4W
                                    Bxxۓ�  
�          @�33�ȣ�?�  @h��A홚C(�3�ȣ׼��
@vffA��RC45�                                    Bxxۢx  "          @�33��33?�@u�A��C&���33=�\)@��\B�C3L�                                    Bxx۱  �          @�  ��=q@ ��@\��A��C!����=q?E�@z�HB��C,��                                    Bxxۿ�  T          @�\�ə�?�G�@X��A�33C&��ə�>��R@l��A�33C1.                                    Bxx��j  
�          @�=q����?��\@UA؏\C(����=�G�@dz�A��C3
=                                    Bxx��  
�          @陚�أ׾L��@+�A���C5���أ׿��@{A�ffC<�                                    Bxx��  �          @�(���G���
=@4z�A�Q�C7�=��G�����@!�A���C>�q                                    Bxx��\  "          @�
=��
=�.{@!G�A���C9����
=��(�@
=qA�p�C@O\                                    Bxx�	  �          @��
��G���G�?�A1p�CB����G���?�R@��CEaH                                    Bxx��  �          @�\)������>�@�Q�CE#�������\)�:�HCE��                                    Bxx�&N  �          @�������0��>���@��
C;&f����G�>��?˅C<
                                    Bxx�4�  T          @�G����R�k�?c�
Ap�C6Y����R��\?E�@�33C9�                                    Bxx�C�  T          @�{��\�G�?Y��@��HC:L���\���\?\)@�{C<(�                                    Bxx�R@  �          @����ff<��
?z�H@��C3ٚ��ff����?n{@�(�C6^�                                    Bxx�`�  
�          @�G���
=    ?}p�@���C4���
=���
?p��@��
C6��                                    Bxx�o�  �          @�G���Q�=#�
?(�@�=qC3����Q�#�
?
=@�p�C5:�                                    Bxx�~2  
�          @�33��\=���>�@n{C333��\�u>��H@q�C4p�                                    Bxx܌�  "          @�G���׾k�>���@J=qC5Ǯ��׾�{>��R@�C6�3                                    Bxxܛ~  �          @�����Q�z�u�   C8����Q���u��C8&f                                    Bxxܪ$  �          @�R��ff��  ���
���C6  ��ff�k����ͿG�C5ٚ                                    Bxxܸ�  
�          @��H��R��z�(����z�C=(���R�c�
�}p�����C;\                                    Bxx��p  
�          @�R��Q쿪=q�����  C>Y���Q�h�ÿ�(��5G�C;�                                    Bxx��  "          @�ff��Q쿢�\�����RC=����Q�^�R��33�.{C:��                                    Bxx��  T          @�\��G����Ϳ�G��>=qC>޸��G��O\)��{�k33C:��                                    Bxx��b  T          @�����ÿ��Ϳ�z��Q��C<����ÿ����s�C8G�                                    Bxx�  "          @����  ��ff����c\)C<}q��  ��(������C7�                                     Bxx��  �          @�z��ᙚ�^�R�z����C;��ᙚ�8Q��\)���HC5xR                                    Bxx�T  �          @�  ���
�
=q���R���C8T{���
������*�RC5                                      Bxx�-�  �          @����녿�Ϳ�=q�I�C8n��논����U�C45�                                    Bxx�<�  �          @������p��%����C7
��>�ff�$z����RC0E                                    Bxx�KF  �          @����
�+���H��C9ff���
=�G�� �����C3{                                    Bxx�Y�  T          @�(���ff�(��������C9B���ff=�� ����{C3�                                    Bxx�h�  �          @�����z��ff�,(�����C7�
��z�>���,����\)C0�R                                    Bxx�w8  
�          @�33��33����+����C7E��33>�G��*�H���RC0n                                    Bxx݅�  �          @�=q���#�
�%����C5B���?&ff�   ����C.�=                                    Bxxݔ�  �          @�����þu�O\)��ffC6
=����?E��J=q����C-�                                    Bxxݣ*  T          @����p����]p���p�C5���p�?s33�U���z�C+��                                    Bxxݱ�  �          @�����(��#�
�dz���Q�C4O\��(�?���Z=q��33C*��                                    Bxx��v  �          @�����(�>�Q��L���ͮC0޸��(�?����;�����C(��                                    Bxx��  T          @�Q�����>�\)�HQ����C1������?�(��9������C)ff                                    Bxx���  �          @������?(��3�
���\C.Ǯ����?�Q��\)���HC'��                                    Bxx��h  T          @�Q���
=?z�H�#�
��G�C+����
=?��H�����33C%��                                    Bxx��  �          @�����{?�
=�4z�����C)�f��{?�p�������C"�                                    Bxx�	�  �          @��H�θR>��H�Y���ݮC/�3�θR?��R�Fff��=qC&��                                    Bxx�Z  �          @�R���
>��s�
��=qC/�R���
?�{�^�R��p�C%ٚ                                    Bxx�'   "          @�p��ə�?��\�j�H��C*��ə�@�L(��ͮC!�                                    Bxx�5�  �          @�\)��{@  �C�
��\)C �R��{@Dz��\)���C��                                    Bxx�DL  �          @�p����
?��Z=q�ӅC$�3���
@0  �-p����Ck�                                    Bxx�R�  �          @����{?ٙ��e���HC%0���{@,���:�H���C:�                                    Bxx�a�  �          @�  �θR@\)�C�
��
=C ��θR@C33�  ��  C�q                                    Bxx�p>  �          @����
=@ff�N{����C"��
=@>{�����  C\)                                    Bxx�~�  �          @�  ��z�?�{�^{��=qC#����z�@3�
�0  ��Q�C:�                                    Bxxލ�  "          @�ff��Q�?���l�����C&(���Q�@$z��E��îC��                                    Bxxޜ0  T          @�����?aG�������\C+�
����@33�g
=��
=C!�
                                    Bxxު�  �          @�ff����?��R�k����HC(�H����@G��I���ɮC &f                                    Bxx޹|  T          @���33@�R�&ff��
=C����33@H�ÿ�  �^=qC�3                                    Bxx��"  �          @���ff@h�ÿУ��L(�C�
��ff@}p����p  Cs3                                    Bxx���  "          @�\��@fff��G��>�\C���@x�þ\�?\)C�
                                    Bxx��n  �          @�33�˅@]p���G���\)Cn�˅@g
==L��>�p�Cn                                    Bxx��  �          @�����@3�
���H�XQ�C�����@L(��O\)��33C�3                                    Bxx��  
�          @��H��33@�G��Q����C8R��33@�33>Ǯ@O\)C�=                                    Bxx�`  
�          @������@�
=�L�Ϳ���C�f���@�G�?�z�A3
=C�
                                    Bxx�   �          @�  ���\@��H>aG�?�\C	�R���\@�=q?�\)APz�Cc�                                    Bxx�.�  �          @�
=����@��>L��?��C	W
����@��H?�{AO\)C
�R                                    Bxx�=R  
�          @��H����@���>L��?�ffC)����@���?�=qAG\)C�3                                    Bxx�K�  "          @���G�@�{��z��4(�C����G�@�(�>#�
?�G�C�3                                    Bxx�Z�  "          @�\��G�@����   ���C(���G�@�(����p  C.                                    Bxx�iD  T          @�  ��p�@����
=���C(���p�@���aG����C xR                                    Bxx�w�  �          @�Q��]p�@�z��Q���
=C J=�]p�@�=q�G���B�u�                                    Bxx߆�  
�          @�\)�B�\@�G��c�
�ffB��\�B�\@�(�����ffB�8R                                    Bxxߕ6  �          @�����=q@u�?B�\@���C��=q@\��?���Aj{CaH                                    Bxxߣ�  �          @�ff��33@�
==L��>ǮC5���33@���?�  A (�Cs3                                    Bxx߲�  �          @�\)��  @��ͿL����33C
����  @�ff>��H@�C
Y�                                    Bxx��(  �          @��
���@�����(��#�C�f���@�
==L��>�G�C�\                                    Bxx���  �          @�=q���R@�ff�333��33CE���R@��R?��@��\C.                                    Bxx��t  �          @�Q����@����$z�����CE���@�녿����\)Cs3                                    Bxx��  "          @�=q����@��R�   �w\)C�q����@��\���|(�C��                                    Bxx���  T          @������@���6ff����C�����@��R��{�4z�B�B�                                    Bxx�
f  T          @�{���R�333@�=qB.��C<�R���R���@|(�B�CK�                                    Bxx�  �          @�\)��{>�  @�ffB'�C0�R��{�p��@�33B"G�C?Y�                                    Bxx�'�  T          @����G�?��\@U�B�C%33��G�>�\)@c33B{C0��                                    Bxx�6X  T          @�p���p�?Y��@s�
BffC*:���p���  @x��BQ�C6��                                    Bxx�D�  �          @�=q�a�@z=q@Q�A�=qC��a�@J=q@H��B(�C
&f                                    Bxx�S�  
�          @�  ����@.�R@2�\A�z�C������?�@\(�B��C                                    Bxx�bJ  "          @�33�{@�{@<(�A�{B뙚�{@l(�@�(�B+B��R                                    Bxx�p�  "          @ƸR���H@��@\)B#  B�ff���H@A�@���B]\)B�                                    Bxx��  
�          @�  ��p�@�  @xQ�B�HB�33��p�@_\)@�G�BM�HB�\                                    Bxx��<  �          @�=q�G�@qG�@^{BQ�C�\�G�@*�H@�(�B;p�Cu�                                    Bxx���  
�          @��H�$z�@�@k�B��B�(��$z�@@  @�{BG�C�{                                    Bxxૈ  �          @��u@�Q�@{�B�HB��f�u@_\)@��\BZ(�B��)                                    Bxx�.  �          @�Q쿦ff@��@tz�B\)Bә���ff@XQ�@�{BW
=B�33                                    Bxx���  T          @�����  @R�\@)��A�{CG���  @(�@]p�B�RC33                                    Bxx��z  �          @�
=���@���@?\)A��B�G����@���@�  B.��B�z�                                    Bxx��   
�          @�\)���
@�
=@�A�G�B������
@��
@s33B
=B��f                                    Bxx���  
�          @��
��p�@�G�@33A�(�B�
=��p�@�{@s33B��B�                                    Bxx�l  T          @��ÿ�(�@���@�A�
=B��
��(�@��@dz�B(�B�p�                                    Bxx�  
�          @��H����@�@�A�33BО�����@�z�@eBffB�                                      Bxx� �  �          @�
=�i��@�G�@(Q�A�  Cn�i��@Z=q@l(�B�C�                                    Bxx�/^  �          @����S33@�  @I��A�\B���S33@O\)@�B(\)C��                                    Bxx�>  
�          @��j=q@���@G�A�{C���j=q@Q�@��B!
=C
+�                                    Bxx�L�  �          @�\)�s33@��H@UA�
=C��s33@B�\@�=qB&=qCff                                    Bxx�[P  �          @Ϯ���@^{@Z=qA��C�����@�@�
=B#=qC.                                    Bxx�i�  �          @˅�=p�@S33@��B5ffC޸�=p�?�p�@��RB^��CB�                                    Bxx�x�  "          @�(����@E@�\)Bb=qB����?�=q@��
B���CG�                                    Bxx�B  
�          @��H�L��@>�R@�=qBn  B��L��?\@�B�  B�u�                                    Bxx��  �          @�{�333@p  @�(�BC��B�B��333@��@�
=B�B�{                                    Bxxᤎ  
�          @�G��8Q�@n{@�  BO�RB����8Q�@z�@�=qB���B���                                    Bxx�4  �          @�  ���@hQ�@���BR�HB��׾��@�R@��B�33B��=                                    Bxx���  �          @���0��@{�@�ffBF�HB��)�0��@"�\@��\B�z�B�G�                                    Bxx�Ѐ  "          @��ÿfff@���@��
B=�B�aH�fff@1�@���Bx�HB׸R                                    Bxx��&  �          @�(��J=q@Q�@��B�{B��)�J=q?��@��B��RC�                                    Bxx���  
�          @��Ϳ�{@Z=q@�ffBX=qB�𤿎{?��R@�p�B�G�B�33                                    Bxx��r  T          @�=q�@`��@�\)B>�B���@p�@��Bo�C}q                                    Bxx�  T          @�
=�8Q�@��R?��AFffB��
�8Q�@�{@(Q�A�z�B�#�                                    Bxx��  �          @\�{@���������B�\)�{@��׿�ff�#�B�8R                                    Bxx�(d  �          @�����
?������
=B�G����
@G
=�����Rp�B�R                                    Bxx�7
  
�          @Å��@������
=B�ff��@�Q�xQ���
B��f                                    Bxx�E�  
�          @�33��Q�@��/\)��p�B֞���Q�@��R��p��3\)Bӽq                                    Bxx�TV  "          @��
��G�@���y���#��B�(���G�@�33�*�H����B��f                                    Bxx�b�  T          @�p��p�@���J�H��\)B��)�p�@�Q�޸R�|  B�\                                    Bxx�q�  
�          @�  �˅@�{�j�H��B�uÿ˅@�\)����B�\)                                    Bxx�H  
�          @׮>�Q�@C33�����z
=B�p�>�Q�@�
=����>z�B��R                                    Bxx��  
�          @�p�?���@W
=��{�e��B�k�?���@��R��(��-33B�ff                                    Bxx❔  
�          @��@%@E��33�Y(�BH(�@%@�p����
�((�Bn                                    Bxx�:  �          @�  @-p�@�33��  �6{Bb�@-p�@�  �r�\���Bz�R                                    Bxx��  T          @��?��@�����ff�A
=B���?��@�ff�w���B���                                    Bxx�Ɇ  "          @�ff��\)@��H���R�=qB�\��\)@���<����(�B�#�                                    Bxx��,  �          @�z὏\)@�����
=�!=qB�z὏\)@Ǯ�>�R��=qB�B�                                    Bxx���  
�          @��=�Q�@�����
=�,��B��=�Q�@���R�\���B�L�                                    Bxx��x  T          @��
>k�@�ff����.B���>k�@���Tz����
B��{                                    Bxx�  "          @�z�>�Q�@�=q����B�G�>�Q�@Ǯ�4z���Q�B�k�                                    Bxx��  T          @�=q>�G�@�=q�vff�
=qB��>�G�@˅��
���B�\                                    Bxx�!j  T          @ۅ?�z�@�=q�r�\���B��?�z�@�33�����ffB�Q�                                    Bxx�0  �          @�\?��\@˅�6ff���B�z�?��\@��
��33��B��                                    Bxx�>�  T          @�=q?5@�(��:=q��
=B�\)?5@�����H�=qB�L�                                    Bxx�M\  T          @��
=���@���C33��\)B�(�=���@�
=����.{B�Q�                                    Bxx�\  T          @�G���33@ə��5����\B�Ǯ��33@�녿�z��z�B�B�                                    Bxx�j�  
(          @أ׿�@�=q�����<z�B�.��@ƸR>u@�B�z�                                    Bxx�yN  �          @���.�R@�p�?�p�A0Q�B�z��.�R@�{@&ffA��B뙚                                    Bxx��  
�          @��H�#�
@fff��ff�CB��#�
@����J�H�
B���                                    Bxx㖚  �          @�=q�}p�@}p��W��G�B���}p�@�����Q�B��                                    Bxx�@  �          @љ���Q�@�=q�W
=��{B��
��Q�@���   ���
B�G�                                    Bxx��  "          @����C33@�z῜(��2�\B�  �C33@���>��?���B���                                    Bxx�  
�          @����]p�@�=q�C�
��33B�G��]p�@������B�p�                                    Bxx��2  �          @��H�dz�@�p��5��{B��3�dz�@�\)�����bffB��{                                    Bxx���  
�          @Ӆ���@k��0���ř�C.���@����\�x��C
&f                                    Bxx��~  T          @�33��  @�����R�6{C����  @�=q�L�;�C�3                                    Bxx��$  �          @���^�R@��=u>��B��^�R@��?��AL  B�\)                                    Bxx��  �          @�G����
@��
?�Q�A��C �=���
@��@H��A�C��                                    Bxx�p  �          @��G�@��@��B'ffB�=q�G�@C33@�\)BR��C��                                    Bxx�)  �          @��AG�@�@��B(�RB����AG�@G
=@�G�BT�RC+�                                    Bxx�7�  �          @�{�L(�@�{@��B*�
B��{�L(�@8Q�@��BT��C	�                                    Bxx�Fb  T          @�R�fff@���@���BB�ff�fff@`��@���B6��C��                                    Bxx�U  T          @�ff�s�
@\(�@�=qB,�C	��s�
@
�H@�Q�BM�HCY�                                    Bxx�c�  �          @�G��7
=@�Q�@��RB��B���7
=@fff@��BB�HC k�                                    Bxx�rT  �          @�Q��=p�@���@�(�B�B�\)�=p�@j=q@�p�B>�RC                                      Bxx��  T          @޸R� ��@�(�@z�HB	�B�#�� ��@���@���B:�HB�Ǯ                                    Bxx䏠  T          @��8��@�G�@�p�B8�B���8��@+�@��Bb�C	(�                                    Bxx�F  T          @陚�n�R@QG�@��B<��C
Ǯ�n�R?��@��B\p�C5�                                    Bxx��  
�          @���y��@@  @�=qB=
=CY��y��?��@�z�BYQ�C+�                                    Bxx仒  �          @��H���\@AG�@��RB1=qC�����\?޸R@���BM
=C�f                                    Bxx��8  
�          @׮���@p�@��RB2  C����?�G�@��BH�RC"��                                    Bxx���  �          @ҏ\�G�@���?�(�A���B�R�G�@l��@<(�Bp�B�\                                    Bxx��  �          @�\)>B�\@�������}G�B���>B�\@���u��
B��R                                    Bxx��*  T          @ƸR?Q�@��\�A����HB��=?Q�@��Ϳ�
=�|  B�33                                    Bxx��  �          @��
>Ǯ@�33�=p���{B�p�>Ǯ@�z´p��P��B�{                                    Bxx�v  �          @�
=?0��@�G��S�
����B�aH?0��@��������B��                                    Bxx�"  "          @ҏ\=�Q�@�
=�������\B�aH=�Q�@�\)�����9��B�p�                                    Bxx�0�  �          @�\)�33@��
������Bۨ��33@ə�?��A��B�\                                    Bxx�?h  �          @�\�
=@�(��.{��Q�B�#��
=@�33?fff@�\B�G�                                    Bxx�N  d          @���   @׮�G���33B�z�   @׮?Tz�@�
=B�z�                                    Bxx�\�  "          @�(�����@�(����
�
ffBȊ=����@�{?\)@��RB�aH                                    Bxx�kZ  �          @�G�    @�z��1���33B��    @��
���\�-p�B��                                    Bxx�z   
�          @�{��\)@ə��-p���  B����\)@�Q쿔z���B�                                    Bxx刦  �          @�
=�#�
@����N�R��ffB����#�
@�\)��33�T��B���                                    Bxx�L  
�          @�G�?���@��
��p��'
=B��R?���@�=q�Vff��=qB�k�                                    Bxx��  T          @��?��@�����p��J��B�?��@�p����\��
B��                                    Bxx崘  �          @��?
=q@�  �^{�33B�8R?
=q@���	����p�B��=                                    Bxx��>  T          @ҏ\�+�@���
=��
=B��׿+�@�{�c�
��G�B��f                                    Bxx���  �          @�Q��
=@�(��7���\)B��ÿ�
=@�(���
=�D  Bъ=                                    Bxx���  �          @�=q��p�@ƸR������\B�.��p�@�p�?h��A ��B�ff                                    Bxx��0  T          @�\�#�
@�z�?���A=��B���#�
@˅@@  A�
=B��=                                    Bxx���  
�          @ۅ�G�@�Q�?��
AR{B׽q�G�@��@<(�A�{B�                                    Bxx�|  T          @����@�z�?�\)AuBܸR��@�G�@R�\A�  B��=                                    Bxx�"  "          @ָR��
@�=q?�z�A��RBُ\��
@�
=@P  A�RB�\)                                    Bxx�)�  
�          @Ӆ�:�H@�z�?�33A�\)B����:�H@���@H��A�z�B��                                    Bxx�8n  "          @�\)�XQ�@�@33A�{B����XQ�@�=q@O\)A�z�B��                                    Bxx�G  �          @�p��\(�@�
=@Q�A�  B�G��\(�@���@dz�A��
B�k�                                    Bxx�U�  �          @߮�:=q@���@7�A¸RB��:=q@��H@��HBz�B�                                      Bxx�d`  �          @߮�P��@�\)@6ffA��RB�p��P��@�{@���B�RB��{                                    Bxx�s  
�          @�=q�J�H@��@EA�B���J�H@�z�@�  Bp�B��q                                    Bxx恬  
�          @�� ��@��H@5�A�(�B��
� ��@�G�@�z�B�B��
                                    Bxx�R  
�          @����1�@�
=@fffA�33B��H�1�@�Q�@�  B'\)B�B�                                    Bxx��  �          @��H�L(�@�ff@�Q�B��B�{�L(�@���@�=qB<��C E                                    Bxx歞  �          @���U�@��@�z�B0  C ��U�@8��@�ffBU�HC�                                    Bxx�D  T          @��R�s�
@@��@��BK�HC�R�s�
?�=q@У�Bf��C}q                                    Bxx���  �          @��u�@tz�@��RB5(�C
=�u�@\)@�BV33C��                                    Bxx�ِ  "          @��R���@K�@�
=B5�RC�)���?�\)@���BO�CǮ                                    Bxx��6  �          @�\)���?˅@�=qBF�\C"����>B�\@ȣ�BOC1��                                    Bxx���  
�          @�ff���\?�G�@��B?�C#h����\>#�
@���BG�HC2.                                    Bxx��  "          @������@&ff@��B?�\C5����?��\@���BTz�C#��                                    Bxx�(  �          @�ff��{?�{@�  B=�CY���{?z�@���BJ�C,��                                    Bxx�"�  "          @�
=���\@\)@�
=B;��C�����\?u@��\BL�HC((�                                    Bxx�1t  �          @�z���@!�@�p�B?��C=q��?��H@��HBS�HC$                                    Bxx�@  �          A (����
@(Q�@�\)BAz�CO\���
?��R@�p�BU=qC%                                      Bxx�N�  T          A����R@-p�@���B@ffC
=���R?�ff@�Q�BTp�C$��                                    Bxx�]f  
�          @�ff��=q@8��@�33B3��C���=q?���@�33BI{C!�R                                    Bxx�l  �          @�z����@>{@��RB9�C�����?У�@�\)BPz�C Y�                                    Bxx�z�  
�          @�=q��  @<(�@�p�B:  C����  ?�\)@�BQ  C :�                                    Bxx�X  T          @�33��33@333@��RB:�RC����33?�(�@�{BP�C"E                                    Bxx��  �          @��|(�@~�R@�(�B$ffC���|(�@333@�z�BE��C�=                                    Bxx禤  �          @���Q�@�33@��B{B���Q�@��@�\)B033B��                                    Bxx�J  "          @��@  @�z�@Y��AمB�\�@  @�  @��B=qB�                                    Bxx���  �          @���HQ�@Å@c�
A�p�B�(��HQ�@�@�=qB�B�33                                    Bxx�Җ  �          @��W�@�ff@^�RA�p�B����W�@���@�ffB��B�k�                                    Bxx��<  �          @����N{@���@b�\A�z�B�Q��N{@��@�{B=qB�\)                                    Bxx���  �          @�{��G�@�33@�B�\CL���G�@^{@�=qB9=qCaH                                    Bxx���  "          @�G��w�@��@�p�B
33B�.�w�@���@�B0�RC�
                                    Bxx�.  	�          @��\�tz�@���@���B ffB�z��tz�@��
@�\)B(�C#�                                    Bxx��  �          @��H�y��@�ff@�=qA�{B�(��y��@�@��B$�CT{                                    Bxx�*z  �          A ���{�@�@�33B+�C33�{�@:�H@�(�BM�\C^�                                    Bxx�9   �          A��x��@P  @θRBOffC��x��?�\@�Q�Bjp�C}q                                    Bxx�G�  �          A
=�^�R@E�@�\)B]�\C
z��^�R?�ff@�By
=C�                                    Bxx�Vl  �          A�R�e@P��@�=qBV33C	���e?�G�@��
Br��C�                                    Bxx�e  "          A�H�s33@8��@�(�BX��C���s33?��@�33Bq(�C޸                                    Bxx�s�  
�          A��i��@Z=q@�z�BO��C�R�i��?���@�\)BmG�C�                                    Bxx�^  �          Ap��i��@U@�(�BP�C	�)�i��?��@�ffBm�C�                                    Bxx�  �          Aff�U�@l��@�ffBQffC���U�@{@�33BrQ�CB�                                    Bxx蟪  "          A Q��{�@AG�@ȣ�BN�\Cu��{�?���@أ�Bg��C��                                    Bxx�P  T          A ���j�H@Tz�@��HBP
=C	�)�j�H?��@��Bl��C�)                                    Bxx��  �          AG��`  @p  @�
=BI�C��`  @�@�(�BjG�CW
                                    Bxx�˜  
�          A��L��@\)@ə�BK�RC ���L��@#33@�Q�Bn��Cu�                                    Bxx��B  �          A33�e@g
=@�p�BN\)C�
�e@	��@�G�Bm(�C\                                    Bxx���  �          A��n�R@\��@��BL�RC	E�n�R@G�@���BiC�)                                    Bxx���  �          A ����
=@w
=@��B1�C	���
=@$z�@��BOC�)                                    Bxx�4  �          A z���p�@�G�@�(�B{C�)��p�@U@�\)B@�HCY�                                    Bxx��  T          A Q��s33@�33@���Bz�C {�s33@i��@�=qBD�C5�                                    Bxx�#�  "          A ����@���@�B��CO\��@XQ�@�G�B8ffC��                                    Bxx�2&  T          A ����ff@{�@�{B*�C
����ff@+�@���BGG�C�R                                    Bxx�@�  �          A   ����@S33@�{B#33C������@�@���B:p�C�                                    Bxx�Or  T          @�ff��ff@���@~�RB �C	�q��ff@Tz�@���BQ�C�3                                    Bxx�^  �          @���z�?�Q�@�\)BOG�C�3��z�>\@�ffBZp�C/                                      Bxx�l�  "          @�33����@ff@��BPz�C������?��
@�33Bc=qC%��                                    Bxx�{d  
�          @�p����R@   @�G�BQ\)C�����R?+�@ʏ\B`  C*�                                    Bxx�
  �          @�����R@  @�  BR��C�����R?aG�@��HBcz�C(&f                                    Bxx阰  �          @���z�?��@�Q�BN(�C���z�?�@ȣ�B[33C,�\                                    Bxx�V  �          @��H���R?�=q@�G�BS
=C�{���R?�\@�G�B_��C-{                                    Bxx��  T          @�\)��
=?�Q�@�\)BK=qC�\��
=?#�
@�  BX��C+��                                    Bxx�Ģ  �          A����R@>{@��\B9G�C�=���R?�
=@�=qBN�C k�                                    Bxx��H  �          A �����H@3�
@�ffB?�C}q���H?�  @��BT{C!�f                                    Bxx���  �          Ap���z�@ff@У�BW\)CY���z�?p��@ۅBi�C'B�                                    Bxx��  
�          A����?���@�(�B[��C�����>Ǯ@ۅBg��C.                                    Bxx��:  T          Ap����R@   @�=qBZ�\C�{���R?z�@�33Bh��C,#�                                    Bxx��  �          A ����p�?���@��B`��CL���p�>��@��HBjz�C1��                                    Bxx��  2          @��\�j�H?G�@�33Bu  C'�3�j�H�(��@ۅBu��C>@                                     Bxx�+,  T          @�Q��Z=q?�(�@ۅBx�C E�Z=q�k�@޸RBQ�C7�                                    Bxx�9�  �          @��;�?�33@��B�C���;����R@�  B�W
C:!H                                    Bxx�Hx  
�          @���?�=q@��B�
=CE��>��
@�G�B���C,�                                    Bxx�W  �          @���X��?��@׮Bs33C&f�X��>L��@�{B��C0��                                    Bxx�e�  �          @���^{?�(�@�z�Bu(�C{�^{    @�G�B~��C3�                                    Bxx�tj  �          @�(��,��?�p�@�p�B���CaH�,��>L��@��
B���C/�                                     Bxx�  T          @�ff��?u@�Q�B��RC8R����@陚B��qCB��                                    Bxxꑶ  �          @���8Q�?�@�ffBw�CB��8Q�?
=q@ָRB���C)T{                                    Bxx�\  �          @�R��@  @�  B�G�C
��?L��@�=qB��3C޸                                    Bxx�  �          @�
=��(�@
=@��
B�\)C��(�?#�
@��B���C!��                                    Bxx꽨  T          @�Q��Y��@%�@�=qBR�C���Y��?�\)@��Bj  C�                                    Bxx��N  �          @���4z�@<(�@��B_=qC�{�4z�?У�@�
=B}{C��                                    Bxx���  �          @���]p�@ff@�=qBY33C�]p�?���@�Bn\)C"Y�                                    Bxx��  �          @�33���@]p�@�(�B
=C�H���@�R@�  B,Q�CaH                                    Bxx��@  �          @�����@���@�(�B=qC33���@Q�@�p�B1=qC�q                                    Bxx��  �          @�33�+�@c33@�=qBR�B���+�@�@�Bup�C��                                    Bxx��  �          @�\)�e@8��@��BI\)C!H�e?�
=@���Bb�
Cٚ                                    Bxx�$2  �          @�
=�h��@R�\@��HBB��C	�H�h��@z�@���B^�HC^�                                    Bxx�2�  �          @�p��s�
@R�\@�B<��C(��s�
@
=@��BW��C�                                    Bxx�A~  T          @�33�tz�@Fff@�ffB?�C��tz�?�z�@�
=BY��Cc�                                    Bxx�P$  "          @�=q�e�@qG�@�ffB9��C��e�@$z�@ÅBX��CY�                                    Bxx�^�  �          @����e@�ff@�Q�BB����e@w
=@�B<p�C�                                    Bxx�mp  �          @��
�e�@�ff@�G�B�B���e�@g�@���B?��C�                                     Bxx�|  �          @������@�z��e���p�B�#׿���@��	����33B���                                    Bxx늼            @��׿�ff@�z��r�\��(�B�#׿�ff@����ff���B�B�                                    Bxx�b  �          @�z��\@����hQ�����B�=q��\@�(��
=q�~ffB���                                    Bxx�  �          @��ÿ!G�@��H��=q�$  B��)�!G�@�R>L��?�G�B��3                                    Bxx붮  T          @�\)���@�녿��R�{B�Q���@��>�\)@��B�8R                                    Bxx��T  T          @�(���33@��H>�33@*�HB�{��33@�?���AaG�B���                                    Bxx���  �          @�{�z�@�Q�?�z�Aj=qB��
�z�@�
=@S33A�G�B�ff                                    Bxx��  �          @��\���@�p�@�A��Bх���@љ�@p  A�ffB�33                                    Bxx��F  �          @���&ff@�
=?�\)A�B۳3�&ff@ڏ\@$z�A��HBݸR                                    Bxx���  �          @���A�@�ff?�
=Ad(�B�(��A�@��@S33A��
B��                                    Bxx��  �          @�ff���@�p�?z�@�z�B�
=���@���@�Axz�B�aH                                    Bxx�8  �          @�ff��@߮�+���p�B�\)��@�\)?E�@�{B�ff                                    Bxx�+�  "          @�Q��@�(���  ��p�Bԙ��@�{>��@g�B�\)                                    Bxx�:�  "          @����
@�z�ٙ��`(�B�녿��
@�33���R�#33B�(�                                    Bxx�I*  T          @�=q��@���=p���Q�B�#���@�?z�@�{B�\                                    Bxx�W�  �          @����Mp�@�G�?�\)A��B���Mp�@�{@�A���B���                                    Bxx�fv  "          @���Z�H@���?��A\)B�B��Z�H@�@��A��RB�.                                    Bxx�u  �          @��p�@�G�>�z�@(�B�W
�p�@��H?�\)AX��B�k�                                    Bxx��  �          @���� ��@θR>�@s�
B�u�� ��@�\)?�G�Ak\)B��H                                    Bxx�h  "          @��;�@�z�?�  Ap�B�L��;�@���@G�A��B�3                                    Bxx�  �          @߮�Dz�@��
?�z�Ap�B�G��Dz�@�Q�@Q�A�  B�#�                                    Bxx쯴  
�          @�G��Y��@�  ?�
=AffB�
=�Y��@�(�@�HA���B�                                    Bxx�Z  
�          @���XQ�@ƸR?��AD��B�#��XQ�@���@1G�A��B�                                    Bxx��   d          @�R�HQ�@�(�?k�@�(�B�B��HQ�@�=q@��A��RBꞸ                                    Bxx�ۦ  "          @�z���  @`  @xQ�B��C+���  @(��@�Q�B#Q�C��                                    Bxx��L  �          @��
��z�@Z�H@l��B�RC����z�@&ff@��B33C�3                                    Bxx���  �          @�  ����@{@��B6z�C� ����?�\)@���BI�C"�R                                    Bxx��  �          @����=q@\(�@�BQ�C�
��=q@!G�@�G�B3�CB�                                    Bxx�>  T          @�{�xQ�@��@0��A���B��f�xQ�@���@n�RB��Ch�                                    Bxx�$�  �          @߮�j=q@�(�@5�A��B�
=�j=q@�ff@s�
B  Cs3                                    Bxx�3�  �          @أ��xQ�@vff@z�HB�\C8R�xQ�@>�R@��
B-p�C�                                     Bxx�B0  �          @�=q��  @aG�@vffB(�Cz���  @*�H@�\)B��C��                                    Bxx�P�  �          @�Q���  @Z=q@�  B��C����  @!�@��B�C)                                    Bxx�_|  �          @�����
@XQ�@���B33C����
@   @�(�Bz�C�R                                    Bxx�n"  �          @�����
@tz�@y��BC
�{���
@<��@�33B$\)C�3                                    Bxx�|�  T          @�{����@^{@��B�
C�f����@{@��RB:p�C�q                                    Bxx�n  �          @����Q�@:�H@��B(z�C���Q�?��@�33B>\)Cc�                                    Bxx�            @�\��  @(Q�@��B  C:���  ?��@���B/Q�C!�=                                    Bxx���  �          @������@U�@��B33C#�����@Q�@���B*(�Cu�                                    Bxx��`  �          @�  ���\?Ǯ@��\BJ  C 8R���\>���@���BT=qC.�                                    Bxx��  �          @�Q���z�?!G�@��BZ��C+Q���z�z�@�p�B[{C;�                                    Bxx�Ԭ  �          @��r�\=��
@���B^p�C2ٚ�r�\���@�BX�\CCk�                                    Bxx��R  �          @���Q�?�=q@��B%�RC'.��Q�>B�\@�{B+�C1��                                    Bxx���  �          @�=q����@Z=q@A�A�{C�\����@.�R@i��BffCz�                                    Bxx� �  �          @�
=��33@�Q�@(��A��C
��33@hQ�@\��A��HC.                                    Bxx�D  �          @����p�@�G�?�Q�A<��C�=��p�@���@�A�C�                                    Bxx��  �          @�\)���@��
?��HA��HC!H���@��
@9��A�p�C5�                                    Bxx�,�  �          @ۅ���
@{�@J=qA��
C�H���
@Mp�@x��B
p�C8R                                    Bxx�;6  �          @��
��
=@��H@-p�A�  C�H��
=@l��@a�A�z�Cff                                    Bxx�I�  �          @�  �tz�@ƸR�޸R�V�\B�.�tz�@���ff�Z�HB�aH                                    Bxx�X�  �          @����G�@Ϯ�<����  B�G��G�@�ff�����:�\B�G�                                    Bxx�g(  �          A z��5@Ϯ�qG���Q�B�B��5@��
�Q���(�B߀                                     Bxx�u�  �          @�{�L(�@θR�I������B���L(�@޸R���
�QG�B�L�                                    Bxx�t  �          @�=q�e�@]p�@��B;33C�q�e�@z�@���BXp�C�                                    Bxx�  �          @�=q����@l(�@�  B%��C^�����@&ff@���BAQ�C�{                                    Bxx��  �          @��R����@}p�@��B�RC
�)����@9��@���B6�HC\)                                    Bxx�f  �          @��R���\@_\)@��
B&�\C�3���\@Q�@�
=B?�RC��                                    Bxx�  �          @�G����R@fff@��B!�C�)���R@   @�B;Q�C�                                    Bxx�Ͳ  �          @�33��{@^{@��B�RC@ ��{@(�@��HB+�
Cٚ                                    Bxx��X  �          A(��A�@��@�ffBD��B�  �A�@C�
@޸RBi  C�3                                    Bxx���  �          A��Dz�@�  @�(�BNB����Dz�@4z�@�Bq�C	c�                                    Bxx���  �          A	�l��@���@�(�BI�C���l��@&ff@��Bi
=Cٚ                                    Bxx�J  �          A\)��{@��@U�A̸RB�
=��{@���@�z�B
�CT{                                    Bxx��  �          A �����H@ۅ?h��@�  B�����H@���@�A��
B��                                    Bxx�%�  �          Ap����H@�=q?�(�A'�
B��H���H@�(�@4z�A��B�L�                                    Bxx�4<  �          A������@��?�{@���B�q����@�{@!G�A��\B��                                    Bxx�B�  �          A(���\)@�ff?���@�{B�p���\)@ҏ\@!G�A�Q�B�k�                                    Bxx�Q�  �          A�����@�\)?��
@׮B�.����@��
@�A�{C �=                                    Bxx�`.  �          A	G���  @�=q����k�B���  @�G�?fff@�G�B�
=                                    Bxx�n�  �          Az��}p�@�\����A��B�Ǯ�}p�@�G������  B�p�                                    Bxx�}z  �          A
ff�j=q@�\���H�Q�B癚�j=q@�=q�\�!�B�8R                                    Bxx�   �          A33�L(�@�������p��B��{�L(�A��!G���p�B��                                    Bxx��  �          A
=q�[�@�Q쿴z���HB�R�[�@�z�>L��?���B�
=                                    Bxx�l  �          A���c33@�G������D��B�ff�c33@�Q쾀  ��33B�33                                    Bxx�  �          A\)�R�\@�ff���zffB���R�\@�Q�8Q����B���                                    Bxx�Ƹ  �          A�
�Mp�@�  �ff���\B�L��Mp�@�=q�E���\)B���                                    Bxx��^  �          A��A�@��H�>{��=qB��f�A�@�G�����Bފ=                                    Bxx��  �          A���X��@�{�G�����B�p��X��@��˅�,  B㞸                                    Bxx��  �          A���dz�@�  �Tz����B����dz�@��ÿ����E�B��                                    Bxx�P  �          A���C33@����G����HB���C33@��H�"�\��{B���                                    Bxx��  T          Aff�3�
@�(���=q���
B�  �3�
@��
�8Q���ffB�Ǯ                                    Bxx��  �          A��J=q@�z��~{���HB��)�J=q@���!�����B�R                                    Bxx�-B  �          A�H�9��@�p�����\)B���9��@�z��1�����B�B�                                    Bxx�;�  �          @�ff�Q�@�=q���\�Bڮ�Q�@�(��O\)���
B�k�                                    Bxx�J�  �          @�(��
=@��\��=q�+��B�(��
=@ə����
��z�B���                                    Bxx�Y4  �          @���?:�H@Tz���z��  B�L�?:�H@����=q�P�B�B�                                    Bxx�g�  T          @�\?�=q@Q���\)k�BQ�H?�=q@p�����H�bffB��                                    Bxx�v�  �          A�@(��@"�\�  �RB/@(��@��\����j(�Bj�                                    Bxx��&  �          A�\@.�R@.�R�G���B3��@.�R@�\)�p��d{Bj��                                    Bxx��  �          A@7�@ �����8RB$�@7�@�Q���f�\B`�                                    Bxx�r  �          A�@   @@  �
=ffBH@   @�p������]=qBwff                                    Bxx�  �          A�@0��@Dz����p�B@G�@0��@�Q���
=�Z=qBo��                                    Bxx�  �          A�@C33@2�\�  #�B)�@C33@�\)����\�B^�H                                    Bxx��d  �          A33@J�H@!��  �Bz�@J�H@�\)�����_z�BT�                                    Bxx��
  �          Aff@#33@3�
���fB?  @#33@�
=����_{Bp�H                                    Bxx��  �          A{@%�@K���H8RBL{@%�@�33��33�X�Bx(�                                    Bxx��V  �          A�\@ ��@]p����{�BXG�@ ��@����\)�R��B�                                    Bxx��  �          A��?�@��H�{�tffB���?�@����G�
B�#�                                    Bxx��  T          Az�@��@u���sz�BhG�@��@���z��I(�B�z�                                    Bxx�&H  �          A�@33@���� (��mz�B���@33@��H�޸R�AQ�B�\)                                    Bxx�4�  �          A=q@(��@qG������o
=B\  @(��@�=q��ff�E�
B~p�                                    Bxx�C�  �          A�H@$z�@�33��G��h��Bg�\@$z�@��
�أ��>z�B��
                                    Bxx�R:  �          A�R@�\@������g�
B��H@�\@����Q��;��B��)                                    Bxx�`�  �          A��@z�@��R�����g\)Btff@z�@�ff�Ӆ�<
=B���                                    Bxx�o�  �          A�@*=q@������d  Bjz�@*=q@�z������9�B�u�                                    Bxx�~,  �          AQ�@�H@������\�g�Bq�
@�H@���أ��<\)B��R                                    Bxx��  �          Aff@/\)@{������i��B\ff@/\)@�
=����@G�B}ff                                    Bxx�x  �          A�
@�@��\��z��_Q�Bxp�@�@�=q�У��3��B�p�                                    Bxx�  �          A��@��@����ff�]�B|�@��@Ǯ�љ��1��B���                                    Bxx��  �          Aff@\)@��R�ʏ\�L\)Bs33@\)@�{��Q��!(�B���                                    Bxx��j  �          @���@@�(���{�M�Bq@@�����ff�"ffB�#�                                    Bxx��  �          AQ�@��@������iz�Bu��@��@�p���z��=�B��\                                    Bxx��  �          A�@�R@�p�����B��B�k�@�R@�����{�B���                                    Bxx��\  �          A	p�@I��@�p���\)�D  B`{@I��@�����z�Bx
=                                    Bxx�  �          A33@	��@��
��  �J�B�B�@	��@��
���\���B�8R                                    Bxx��  �          A�
@p�@��Ӆ�D(�B�33@p�@�ff���
��\B�B�                                    Bxx�N  �          A	�@'
=@��H����IBv��@'
=@�z���  �  B��                                    Bxx�-�  �          A
�\@333@�
=���
�Q��Bg@333@�������'\)B��{                                    Bxx�<�  �          A
�R@4z�@�=q��p��?�\Bs�H@4z�@�����R�  B��f                                    Bxx�K@  �          A��@.{@�p������B�
Bt{@.{@�����
=�33B�B�                                    Bxx�Y�  �          A�H?�33@���ٙ��a��B�aH?�33@�ff��\)�233B��                                    Bxx�h�  �          A�H?@  @�{��G��aB�?@  @�����ff�1�B�#�                                    Bxx�w2  �          A
�R?�p�@������^z�B��f?�p�@�����  �.�B�\)                                    Bxx��  �          A
=?O\)@��\����i�B��?O\)@�  �\�9G�B�
=                                    Bxx�~  �          A�>�33@����\)�rG�B�(�>�33@�  �ƸR�A�B��R                                    Bxx�$  �          A�?��@��
���H�b��B�#�?��@�Q���ff�2(�B��{                                    Bxx��  �          A
=q?��R@�G���\�\(�B��)?��R@������,\)B�=q                                    Bxx��p  �          A	��?J=q@�Q����_z�B�33?J=q@����{�.�HB�k�                                    Bxx��  �          A\)>�{@�
=���H�W��B�B�>�{@�����
�&��B��f                                    Bxx�ݼ  �          A��?�@�z���=q�Sp�B�aH?�@�
=��=q�"p�B�                                    Bxx��b  �          A
ff?!G�@��R��(��R��B�#�?!G�@������!�B���                                    Bxx��  �          A�?�
=@�ff��Q��L(�B�=q?�
=@�������B�(�                                    Bxx�	�  �          A
�\?k�@�ff��(��H�B�Ǯ?k�@׮�����HB�k�                                    Bxx�T  �          A
�H?�ff@�  ��=q�E�B�u�?�ff@�������B�Q�                                    Bxx�&�  �          A�H=�Q�@�{������B�z�=�Q�@��
�^�R��ffB���                                    Bxx�5�  �          A=q=�@�z���\)��Q�B�
==�@�(��)����33B�=q                                    Bxx�DF  �          A�R=���@�Q���p��
=B�8R=���@�\)�l������B�u�                                    Bxx�R�  �          A�?xQ�@�p���{�-�
B�p�?xQ�@�Q�������\)B�B�                                    Bxx�a�  �          A��?��@��������"�HB��H?��@����xQ���{B�=q                                    Bxx�p8  �          A�=���@����p�� 
=B�Q�=���@�33�2�\��ffB�z�                                    Bxx�~�  �          @�
=�!G�@�z�>�Q�@)��Bٔ{�!G�@�z�?���Ag
=B���                                    Bxx�  �          A�=�Q�@�ff�x������B��{=�Q�@�(��\)�~{B��3                                    Bxx�*  w          A	p�?u@������
B���?u@�p��o\)��z�B��3                                    Bxx��  	�          Az�?��@��H��ff�!�B�.?��@�p���z�����B�Ǯ                                    Bxx�v  �          A33?���@Ǯ�����+{B�{?���@��
������B��                                    Bxx��  T          A�?��@�33���H�%Q�B�G�?��@�p���(���B���                                    Bxx���  �          A  ?�33@��H��Q��'��B���?�33@���������B�8R                                    Bxx��h  �          A��?���@�{��(��?��B��q?���@�����33��HB�{                                    Bxx��  �          A(�@{@�33������B���@{@���g
=��  B�\                                    Bxx��  "          A
=?�z�@�33��G��4G�B��3?�z�@�\)���
=B�L�                                    Bxx�Z  
�          Ap�?��@�����33�QG�B�ff?��@�\)���
� 33B��f                                    Bxx�    w          Az�?�=q@�ff�ָR�Y33B�=q?�=q@�������'�B��H                                    Bxx�.�  
�          A=q?�@n�R��Q��n�HB��)?�@���Q��?ffB�Q�                                    Bxx�=L  "          @���?G�@�ff��ff�\
=B�8R?G�@�  ����)��B���                                    Bxx�K�  T          @��R?�(�@i�����kp�B���?�(�@������R�<(�B�L�                                    Bxx�Z�  "          AG�?xQ�@!G���=q�3B�\?xQ�@�(��ڏ\�h�B���                                    Bxx�i>  �          @���?���>�����H�Ag�
?���@z���=q�HBS�                                    Bxx�w�  1          @��?��׽L����
=�RC��?���?�z���G�BI(�                                    Bxx�  �          AQ�?���@�����p��h�B��?���@�  �����6�B���                                    Bxx��0  �          A�?���@�Q����b  B��?���@�(���=q�0(�B���                                    Bxx���  E          Aff?��\@�Q����
�d�B��H?��\@���  �2�RB��H                                    Bxx��|  c          A  ?��@�G����H�7  B��?��@�  ��z���\B�(�                                    Bxx��"  �          A�\?��
@�=q��33�:=qB�(�?��
@�G���ff�=qB��                                    Bxx���  "          A��?���@�p���  �)G�B�Q�?���@陚��\)��\)B�
=                                    Bxx��n  �          AQ�?�p�@��Å�8�\B���?�p�@����{�=qB���                                    Bxx��  �          A33?�G�@�{��(��!��B���?�G�@�G���G���  B�\)                                    Bxx���  "          A��?�
=@���ҏ\�BG�B�u�?�
=@�������(�B��                                    Bxx�
`  �          A��?�@�\)�����L��B��H?�@�=q�����
B�\)                                    Bxx�  �          A
=?�
=@�p������M33B��=?�
=@�\)���
���B��                                    Bxx�'�  �          Ap�?�=q@��\�ۅ�h
=B���?�=q@��������5��B���                                    Bxx�6R  �          @�
=?��
@r�\��z��o=qB���?��
@�  ��33�=
=B��                                    Bxx�D�  �          A�?��@�Q����jG�B���?��@�\)���H�7��B�k�                                    Bxx�S�  �          A�
?�z�@j=q�����xB��H?�z�@��R��Q��F�B���                                    Bxx�bD  �          A�?�{@-p���z���B}\)?�{@���=q�c  B��)                                    Bxx�p�  "          AQ�?��@&ff� z�B���?��@����
=�h=qB���                                    Bxx��  �          A	�?(�@1��G�=qB�W
?(�@�G���\)�f�\B�Ǯ                                    Bxx��6  �          A��?��@/\)����Q�B�\)?��@�{��{�d��B���                                    Bxx���  �          @�?��?�\)���
�fB�L�?��@P�����33B�Ǯ                                    Bxx���  
�          A�?���@�=q���H�XQ�B�ff?���@�ff���
�%(�B��                                    Bxx��(  "          A33?�p�@����Ӆ�M�RB�B�?�p�@������ffB�                                    Bxx���  �          A\)?�=q@�����Q��TffB���?�=q@�=q���R� �RB�=q                                    Bxx��t  �          A��?�Q�@l(����H�xB�Q�?�Q�@����ə��E\)B���                                    Bxx��  T          A��?�\)@_\)��\)���B�=q?�\)@��
��
=�L
=B�33                                    Bxx���  �          A��?�=q?���   ��Bq  ?�=q@k����H�yB�aH                                    Bxx�f  
Z          AQ�?G�@   ��
=�fB�aH?G�@q���G��xG�B��q                                    Bxx�  �          Aff?��
@x���陚�rB�� ?��
@�\)��ff�?�B���                                    Bxx� �  �          A	G�?�{A=q���m�B�  ?�{A�H�k�����B���                                    Bxx�/X  "          A
=?��R@��
�)������B��{?��RAQ�@  ���B���                                    Bxx�=�  �          A�H?���@�����  ���B��?���@��\�   ��  B�k�                                    Bxx�L�  �          AG�@
=q@�33��z���ffB�=q@
=qA�R�#33��ffB�aH                                    Bxx�[J  �          AQ�@�@�p����\��Q�B��
@�Az��.{��ffB�8R                                    Bxx�i�  "          AG�@ff@�=q���H� �HB�ff@ffA  �@  ��  B�33                                    Bxx�x�  T          A
=@{@�����R�  B�(�@{@��H�^{���B�{                                    Bxx��<  T          A��@��@�p���
=�z�B�u�@��@��R�`����B�p�                                    Bxx���  T          A33@��@ʏ\��ff�\)B�k�@��@�{�s�
���
B�.                                    Bxx���  �          A�@��@�(������(�B��@��@�  �w
=��=qB�Q�                                    Bxx��.  �          A��@ff@�G���{�&�\B���@ff@�R��33���HB�z�                                    Bxx���  �          A
=@�@�z��ƸR�7  B��@�@�ff��ff�=qB��                                    Bxx��z  �          Ap�@��@��\��\)�3�B��\@��@�z���p���\)B���                                    Bxx��   �          A�@!�@��H��=q�3�B�=q@!�@���  ���B�8R                                    Bxx���  
�          A�@�R@��������:{B�  @�R@�Q����
�{B�Ǯ                                    Bxx��l  T          A@�R@�=q�����#�B�(�@�R@�����
���B�                                      Bxx�  
�          A�R@(�@Ӆ�����B�G�@(�@���tz���33B�aH                                    Bxx��  �          Ap�?˅@����  ��\B�aH?˅A��7
=���\B�{                                    Bxx�(^  
�          A�R?�33@ƸR��=q�33B��?�33@���j�H��p�B���                                    Bxx�7  T          A(�?�(�@��
���\�2��B�  ?�(�@ۅ�������B��)                                    Bxx�E�  T          A   @G�@��\��33�>z�B��q@G�@�33��Q��  B�8R                                    Bxx�TP  "          @�
=?���@�G���  �^�B���?���@�����\�*��B�Ǯ                                    Bxx�b�  E          @���@;�@-p���G��n=qB+ff@;�@������C�B^z�                                    Bxx�q�  1          A��?(��@ָR��G���\B��?(��@�p��AG���p�B��                                    Bxx��B  �          A��>�
=@���������B�B�>�
=@���H��\)B���                                    Bxx���  E          A��?B�\@ҏ\��=q���B��)?B�\@��
�Tz���\)B���                                    Bxx���  
�          Az�?0��@�33����<Q�B�
=?0��@�������=qB���                                    Bxx��4  
i          A\)?�
=@�(���Q��*Q�B�8R?�
=@ᙚ�z=q��\)B��                                    Bxx���  T          Aff?��R@�����{�j\)B�
=?��R@�(�����3z�B�33                                    Bxx�ɀ  "          A�R?޸R@}p����h  B���?޸R@�G�����2�RB�#�                                    Bxx��&  �          A�H@   @�33����?ffB�L�@   @�{���
�	�B��=                                    Bxx���  w          A ��?��
@�\)��
=�A�HB���?��
@ə������\B�aH                                    Bxx��r  
�          @��?�Q�@�(�����3�B���?�Q�@����xQ����\B�(�                                    Bxx�  �          @��?aG�@\����p��k�B�� ?aG�@�����(��3��B�W
                                    Bxx��  �          @�?��@�ff�����+�B��{?��@��H�:�H��ffB�#�                                    Bxx�!d  �          @�z�>#�
@ʏ\?�33A���B��>#�
@��@Z=qA�z�B�Ǯ                                    Bxx�0
  
�          @�  >�Q�@��L�;��HB��>�Q�@Ǯ?�ffA^{B�Q�                                    Bxx�>�  T          @�Q�?B�\@�=q�����AQ�B��q?B�\@�33�_\)�ffB���                                    Bxx�MV  T          @��?p��@�\)�333�Ə\B�\?p��@Ϯ��(��(z�B�u�                                    Bxx�[�  �          @�?B�\@�{���33B�?B�\@�G��   ��G�B��                                    Bxx�j�  f          @�(�?���@������H���B�W
?���@ٙ��O\)�ə�B��\                                    Bxx�yH  �          @�z�?W
=@�p���G��:p�B���?W
=@�����=q� �HB�{                                    Bxx���  �          @�=q?���@�Q����R�D\)B�k�?���@�G����\���B��H                                    Bxx���  �          @�\)?h��@�(���p��(�
B��\?h��@�  �e��{B��{                                    Bxx��:  �          @�z�>�@�{���{B�8R>�@��.�R��\)B�\)                                    Bxx���  �          @�z�>B�\@�=q��=q�B�k�>B�\@����8Q���=qB��                                    Bxx�  �          @��>�@�Q������HB��>�@���N{��=qB��f                                    Bxx��,  �          @��H�Ǯ@�{��Q��J  B��þǮ@ə�����=qB�                                    Bxx���  �          @�Q��@�{�O\)��(�B�B���@陚��
=�0��B��{                                    Bxx��x  �          @�
=�p��@����o\)��{B��Ϳp��@���������B��
                                    Bxx��  �          @�z�Q�@�33�_\)��RB�ff�Q�@�G������n=qB��
                                    Bxx��  �          @�ff�(�@�(�����B��(�@ȣ��*=q���HB�.                                    Bxx�j  �          @���  @��R�z�H��
B��ÿ�  @����!G���ffBը�                                    Bxx�)  �          @����=q@������\�  B�{��=q@���)������B�.                                    Bxx�7�  �          @��\)@�  �=p�����B��=�\)@ٙ���  �&�\B�W
                                    Bxx�F\  �          @�Q�\)@�����(���B�\�\)@���(Q����\B�W
                                    Bxx�U  �          @�ff��@�ff�{��	�B�p���@У��z���(�B��                                    Bxx�c�  �          @�33�c�
@���W
=��
=B�p��c�
@�=q��33�W�B��)                                    Bxx�rN  �          @أ׿�(�@�  �#33��  B��῜(�@ƸR�xQ���B�=q                                    Bxx���  �          @�z�s33@��\�����B�(��s33@��?E�A��B�G�                                    Bxx���  �          @�p���@�33����4��B�B���@���S33���B���                                    Bxx��@  �          @�
=>�
=@��H��G��
=B�  >�
=@�
=�   ����B�B�                                    Bxx���  �          @ٙ��^�R@�33�������B�aH�^�R@�
=�\)����B��
                                    Bxx���  S          @��Ϳ���@���mp���
Bʳ3����@�z��
=��B�                                      Bxx��2  
�          @�?�{@�Q��z=q���B��?�{@��H�����G�B���                                    Bxx���  �          @�\)?�G�@�����  �B�8R?�G�@У��8Q�����B�W
                                    Bxx��~  �          @�
=����@�{�U���
B�k�����@��
��(��nffB�{                                    Bxx��$  �          @�����
@�G�����r�\B�\)��
@�G��u��B�
=                                    Bxx��  �          @�(��G�@�z��3�
����Bܔ{�G�@�p������
�RBٞ�                                    Bxx�p  T          @�p���Q�@�p����
�!�RB�k���Q�@���2�\�У�B��
                                    Bxx�"  �          @�33���@O\)����j��B�W
���@��
�����/  BΨ�                                    Bxx�0�  �          @�\)��  @_\)���
�X{B�.��  @��������B�                                      Bxx�?b  �          @�(��\)@���p  �{B�u��\)@�{��
����B���                                    Bxx�N  �          @�
=��33@n{��\)�G�B�LͿ�33@�33�c33�z�B�=q                                    Bxx�\�  �          @�
=��@��R�j�H�z�B�G���@�  �Q���{BՅ                                    Bxx�kT  T          @�p��\)@�ff���
=B��f�\)@����0�����B�.                                    Bxx�y�  �          @Ӆ�K�@�z��G���33B�
=�K�@�녿�
=�k�
B�{                                    Bxx���  T          @��p��@����<(���=qC�3�p��@�������ap�B�33                                    Bxx��F  �          @Ӆ��z�@6ff�%��ffC!H��z�@\(���(��q�Cff                                    Bxx���  �          @ٙ���{@w
=�!G���\)C���{@��Ϳ����<(�C
Q�                                    Bxx���  �          @�ff���H@��?xQ�@��C�����H@�{@A��C	L�                                    Bxx��8  �          @�\�z=q@�Q�@!�A���Cc��z=q@x��@o\)B�C&f                                    Bxx���  �          @�=q�I��@��H@K�A�B�\�I��@���@�ffB$(�C {                                    Bxx���  �          @�G����@��\?n{@��C����@�z�@p�A�p�C�f                                    Bxx��*  T          @�����@����p��B�\CW
����@���?G�@ϮC��                                    Bxx���  �          @�
=��@�=q?�\)A8��C#���@�  @.{A�ffC�{                                    Bxx�v  �          @��H���@�(�?�\Ah��C:����@�{@G�A�=qCY�                                    Bxx�  �          @�
=����@��\@'
=A�ffC	������@\(�@mp�B �C+�                                    Bxx�)�  �          @�������@qG�@6ffA�\)C)����@5�@q�B(�C�
                                    Bxx�8h  �          @޸R����@:�H@3�
A��CG�����@G�@`��A��
C                                       Bxx�G  T          @�z���  @8Q�@]p�A��C^���  ?�ff@��
B��C!޸                                    Bxx�U�  �          @�R��p�?�ff@hQ�A��HC#{��p�?8Q�@~�RB��C-
                                    Bxx�dZ  �          @�����{>�p�@AG�A��HC0����{���H@@  AǙ�C8Y�                                    Bxx�s   �          @�G���Q�@�@h��A���C5���Q�?��@�p�Bp�C&��                                    Bxx���  �          @�\��\)?�G�@q�B{C%8R��\)>Ǯ@���B��C0�                                    Bxx��L  �          @�Q���33?���@�{B{C(0���33�#�
@�=qB �C5�q                                    Bxx���  �          @ָR��33?��@�ffBKffC%�\��33���@���BO�HC:�\                                    Bxx���  �          @�����\?��@�Q�B�C$����\>L��@��B�C1�R                                    Bxx��>  �          @�33��(�?u@�\)B!p�C(����(���\)@��\B&  C7E                                    Bxx���  �          @�33��?(��@z=qB(�C,� ����@{�BQ�C9E                                    Bxx�ي  �          @�z���Q�=#�
@���B��C3�=��Q쿚�H@�\)BG�C@��                                    Bxx��0  �          @�\)��G���
=@�  B��CA0���G��ff@o\)B�CL��                                    Bxx���  �          @˅��p��G�@}p�B\)CO5���p��S�
@I��A홚CXٚ                                    Bxx�|  �          @ʏ\��G���G�@�G�B7Q�CAL���G��G�@��B�\CO��                                    Bxx�"  �          @����(�>\)@�=qBffC2L���(����
@|(�B\)C?�                                    Bxx�"�  �          @����33?�@vffB��C&h���33<#�
@���BC3�{                                    Bxx�1n  �          @˅��\)?��\@���B=qC$�R��\)=L��@��B'z�C3ff                                    Bxx�@            @�(����?�@l��Bp�C8R���?�@���B(�RC-�                                    Bxx�N�  �          @�{��=q?��@Z=qBQ�C!=q��=q?z�@o\)Bp�C-(�                                    Bxx�]`  �          @ə�����@G�@'�A�p�C8R����?�z�@J=qA��C$��                                    Bxx�l  �          @θR��녿Ǯ@���B��RCd(�����O\)@��Ba�Cv��                                    Bxx�z�  �          @����p��?�z�@�ffBb  C"���p�׿!G�@���Bf�C=u�                                    Bxx��R  �          @�p����H@�G�@��
BOffB�W
���H@33@�=qB�W
B뙚                                    Bxx���  �          @���xQ�?˅@s�
B)
=C��xQ�>���@�33B9�HC.)                                    Bxx���  �          @Ϯ�?\)@�z�@5�A�\)B��=�?\)@fff@�=qB$  C�f                                    Bxx��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxx�Ґ   @          @��ͿTz�@߮?��HA�B�� �Tz�@ʏ\@N{A�Q�B��                                    Bxx��6  �          @ᙚ�Y��@�ff>�z�@��B��H�Y��@љ�@A��B¸R                                    Bxx���  �          @�\)�=p�@��=�?s33B�(��=p�@��@
=qA���B�Ǯ                                    Bxx���  �          @�zῈ��@�p�?fff@�=qB�LͿ���@�33@5A��B���                                    Bxx�(  �          @ٙ���=q@ə�?\)@��
B�aH��=q@�33@=qA��
Bֽq                                    Bxx��  �          @޸R�G�@�ff����p�B����G�@��?�A?�Bۏ\                                    Bxx�*t  �          @���u@�33?�Q�A�G�B��ÿu@��@j�HB
\)B��
                                    Bxx�9  �          @�(��ٙ�@�G�@%A�BԮ�ٙ�@�  @��RB\)B�\)                                    Bxx�G�  �          @ָR�Q�@���@�A��HB���Q�@���@h��B	33B��                                    Bxx�Vf  �          @�  �5@��@1�A�33B�Ǯ�5@���@���B�B�.                                    Bxx�e  �          @�{���@�@�Q�Bo  C����?�\@�ffB�W
C'G�                                    Bxx�s�  �          @��
�*�H@z=q@��B%=qB��=�*�H@{@���B[(�C	0�                                    Bxx��X  T          @����(�?O\)@�z�B%�\C*���(���@�ffB({C9��                                    Bxx���  �          @����Q�>#�
@���BH\)C1����Q쿧�@��
B>p�CF                                    Bxx���  �          @���!G�@<(�@��HBb��B�{�!G�?�33@�Q�B�k�B��                                    Bxx��J  �          @�  ��  @ff@XQ�BD(�B�W
��  ?��H@|(�BvffCff                                    Bxx���  �          @����@��@Dz�A�C����?��H@g
=B
��C&�                                    Bxx�˖  �          @�(����
@ff@j=qB{CT{���
?�\)@��RB
=C'��                                    Bxx��<  �          @�����z�@:�H@5A�z�C����z�?�33@eA�
=C!T{                                    Bxx���  �          @�\����@>�R@P��A���C
����?�=q@���B�RC!@                                     Bxx���  �          @������H@��\@�A��C�H���H@W�@h��A��RC�                                    Bxx�.  �          @��
��33@���@33A�(�C\��33@�p�@aG�A�(�C��                                    Bxx��  �          @�ff�p  @��R?�=qA\��B�k��p  @�
=@G�A���C�q                                    Bxx�#z  �          @����P  @�
=?�Q�A$Q�B�(��P  @�=q@:=qA�z�B�L�                                    Bxx�2   �          @��Ϳ��
@�Q��0����B�(����
@�33�fff�  B˸R                                    Bxx�@�  �          @�z�
=@��R�����F��B���
=@�=q�\�����B��                                    Bxx�Ol  �          @�p�>�
=@�p�����!��B�>�
=@�G��%���B�ff                                    Bxx�^  �          @�?���@�����Q��9�B�=q?���@�p��I����z�B��                                     Bxx�l�  �          @�=q��33@Å@5A�33B���33@��@��
B(�\B�Ǯ                                    Bxx�{^  �          @�Q�>u@ָR��(��hQ�B���>u@�  ?��HAk\)B���                                    Bxx��  �          @��>�{@�{�Tz����B�\)>�{@��H?�33A8��B�G�                                    Bxx���  �          @��H>��@���?
=@��B��{>��@˅@/\)A���B�aH                                    Bxx��P  �          @��þ\@��?ٙ�Ah  B����\@�{@l(�B�RB���                                    Bxx���  �          @�ff?0��@�(���  �B��\?0��@�33?�Q�A��\B��                                    Bxx�Ĝ  �          @�
=��G�@���?��HA��B�=q��G�@�\)@i��B�HB��                                    Bxx��B  �          @��>{�%�@��
BR{C\�R�>{��=q@���B�RCi�3                                    Bxx���  �          @�(��\@�
=@Q�B(�B���\@j=q@�ffBGp�B�\                                    Bxx���  �          @�녿��R@�=q@�z�B��Bυ���R@�@��BM��B�#�                                    Bxx��4  �          @����33@��@L(�A�33B�Q쿳33@�p�@�ffB+�\B�
=                                    Bxx��  �          @�\�2�\@���@�33B33B�=�2�\@u@�z�BL��B�                                      Bxx��  �          @����ff@�G�@��B)��B�G��ff@J�H@�33Bl\)B�33                                    Bxx�+&  �          @���+�@��
@�Q�B233B�33�+�@+�@�
=Bn��C�                                    Bxx�9�  �          @��#�
@ۅ?�  AW�B��)�#�
@�p�@xQ�A�(�B�                                    Bxx�Hr  �          @�R��@33@��
B��C#׿���z�@��B�ffC=�\                                    Bxx�W  �          @�����@*�H@�ffB|�C ���>�33@�RB��qC*�                                     Bxx�e�  �          @����R@0��@љ�Bq��C����R>��@��HB���C)+�                                    Bxx�td  �          @���:=q@u@��HBN��B�L��:=q?��@�Q�B�C�=                                    Bxx��
  �          @��=q@?\)@�{Bl�C ���=q?5@�=qB���C#z�                                    Bxx���  T          @�R�33?�@�Q�B��HCL��33�u@�\B���CM�                                    Bxx��V  �          @��У�?�G�@�B�� C	@ �У׿Y��@�G�B��RCO�H                                    Bxx���  
�          @�G���  ?�p�@�  B��
Cs3��  ���@��B��)C^Y�                                    Bxx���  
�          @�\)��@e�@��BW�B���?�
=@�Q�B�k�C�)                                    Bxx��H  �          @��
��
@��R@�  B9ffB�8R��
@(Q�@׮Bz\)CJ=                                    Bxx���  �          @��8Q�@�  @��HB8�B�33�8Q�@\)@�ffBq��C{                                    Bxx��  �          @�  �@��@#�
@�  Bf�
C���@��>�{@�  B��C-�                                     Bxx��:  �          @����\@�p�@��B5�B۔{��\@7�@���B|ffB�G�                                    Bxy �  �          @��	��@3�
@�=qBf�B�Ǯ�	��?L��@�B�L�C��                                    Bxy �  �          @�Q��B�\?�(�@љ�BtQ�C��B�\����@ڏ\B�ǮC:G�                                    Bxy $,  �          @����8��?�
=@��HB|{C�3�8�ÿE�@�ffB�33CB�R                                    Bxy 2�  �          @����@�p�@���B\)C�
���@.{@���BB  C+�                                    Bxy Ax  �          @��H�s�
@n{@��RB+�
C���s�
?�@�p�BY�
C@                                     Bxy P  T          @��>W
=@Ӆ@���A�z�B�L�>W
=@��
@�Q�BK�HB���                                    Bxy ^�  �          @��H�Ǯ@��
@u�A�33B�p��Ǯ@�{@��HBAB��                                    Bxy mj  �          @����=q@ָR@n{A���B�uÿ�=q@���@���B=��B֨�                                    Bxy |  �          @�p�����@��H@}p�A�z�B�=q����@��@��RBE�B�ff                                    Bxy ��  �          @�����@�=q@���BG�B�\)���@��@���BW�\B�z�                                    Bxy �\  �          A   �.{@�p�@�Q�B%B�\�.{@q�@޸RBtz�B�W
                                    Bxy �  �          @��
�+�@���@�
=B(�B��+�@~�R@�\)Bl�B�B�                                    Bxy ��  �          @�G��\@��H@p��A�Q�B�녿\@��@���BAffB�Q�                                    Bxy �N  �          @�\)��
=@�Q�@Tz�A�{B��)��
=@��R@�B3�Bר�                                    Bxy ��  �          @��ÿ�Q�@��@   A�  BΏ\��Q�@��
@���B�B��                                    Bxy �  �          @�33��=q@޸R@-p�A��Bɔ{��=q@��@�B"=qBΞ�                                    Bxy �@  �          @��H���H@�p�?�  AV=qB��)���H@�(�@��B��B�8R                                    Bxy ��  �          @�녿���@�p�@S33A�G�BǸR����@��@��B3�HB�L�                                    Bxy�  �          @�
=���
@�G�@+�A��B̊=���
@�p�@�ffB p�B�.                                    Bxy2  T          @��?J=q@ҏ\@���B=qB�ff?J=q@�{@�G�BS�B�                                      Bxy+�  "          @��R?�33@�(�@�  B{B�aH?�33@��@�G�BP�RB��
                                    Bxy:~  T          @�(����R@�\)?��A_
=B�uÿ��R@�33@��
BffB�Q�                                    BxyI$  �          @��
��=q@�33@�
As
=B�  ��=q@�p�@��B�\B�\                                    BxyW�  �          @��;u@�
=@�A��RB��R�u@�{@���B
=B�z�                                    Bxyfp  �          @��
?0��@��@1�A��HB�.?0��@��\@���B$�B��=                                    Bxyu  �          @�33@ ��@��@\)A��
B���@ ��@��\@��HBffB��                                    Bxy��  �          @�=q@��@ۅ@.{A��
B���@��@��R@��RBp�B���                                    Bxy�b  �          @�(�?�Q�@�@��A�33B��?�Q�@���@�=qB33B�G�                                    Bxy�  �          @��
?���@��?��A
=B�.?���@ָR@z=qA�z�B��3                                    Bxy��  �          @��
?�@�(�?�(�A.=qB�B�?�@��
@�=qA�Q�B�{                                    Bxy�T  �          @�(���(�@�
=?�  AM��B��f��(�@˅@���B�B�33                                    Bxy��  �          @��ͿE�@�@"�\A��B���E�@�\)@��BffB½q                                    Bxy۠  �          @�33�aG�@���@G
=A�
=B���aG�@�=q@�ffB0
=B�                                    Bxy�F  T          @�p����
@��@��A���B��H���
@�@��B�B��                                    Bxy��  �          @��;�p�@��H@5�A��\B��q��p�@�33@���B'��B�33                                    Bxy�  �          @�p���z�@�\)@5�A��B�(���z�@��@�\)B'
=B�                                    Bxy8  �          @�\)��z�@�?��
AO\)B��ÿ�z�@�
=@�z�B33B̏\                                    Bxy$�  �          @���k�@���@
�HA�
B�  �k�@Ǯ@�\)B\)B���                                    Bxy3�  �          A ��>���@�\)?8Q�@�(�B���>���@�{@b�\AхB�#�                                    BxyB*  �          A z�>.{@��R?k�@ҏ\B��{>.{@�33@n�RA݅B�G�                                    BxyP�  �          Ap�?E�A zὸQ�&ffB�{?E�@��@3�
A�\)B�\)                                    Bxy_v  �          A ��?�\@���
=���B�?�\@��@A�\)B�u�                                    Bxyn  �          @���?0��@�Q�#�
��G�B���?0��@�\)@333A�ffB�{                                    Bxy|�  �          @��H?@  @��þ��VffB�  ?@  @�p�@��A��RB�u�                                    Bxy�h  �          @���=�G�@�=q=�?c�
B�\)=�G�@�@=p�A�B�=q                                    Bxy�  T          @�p�>���@���?�33A#�B��>���@�
=@�z�A��\B�.                                    Bxy��  T          AG��   @�
=?�33A33B�(��   @�  @~�RA��B�#�                                    Bxy�Z  T          A �ýL��@�ff?��\@�\B��R�L��@���@w
=A�Q�B���                                    Bxy�   �          A �׿   @��?!G�@�
=B���   @�ff@`��A�p�B��H                                    BxyԦ  �          Ap��L��A z�>�(�@C33B�p��L��@��@VffA��B��\                                    Bxy�L  �          @�33��R@�=q>\)?��
B���R@�
=@@��A�ffB�                                    Bxy��  �          @�  �(�@��R>�Q�@(Q�B���(�@�G�@J�HA��B���                                    Bxy �  �          @�(��z�H@���>W
=?�G�B�Q�z�H@�@Dz�A�  BÏ\                                    Bxy>  �          @��R�333@�p�=#�
>���B�(��333@��H@>{A�p�B��                                    Bxy�  �          @�
=�O\)@���L�Ϳ���B����O\)@�p�@0  A���B�aH                                    Bxy,�  �          @�p��(�@��ÿ�����B��(�@�z�?�=qAV�RB�{                                    Bxy;0  �          @�ff�:�H@����=q�7\)B��:�H@���?�A$��B��R                                    BxyI�  �          @��ÿ���@�33������B��ῌ��@�z�?333@�B�W
                                    BxyX|  �          @�ff��@�z��
=��\)B����@�G�>�z�@�
B�                                    Bxyg"  T          @�  � ��@陚��G��G�B�� ��@��@(Q�A��B܊=                                    Bxyu�  �          @���:=q@��Ϳ&ff����B�8R�:=q@��
@�Ayp�B��
                                    Bxy�n  �          @�{�;�@�=q?W
=@�G�Bߨ��;�@�
=@aG�A�
=B�q                                    Bxy�  �          @�ff�6ff@�{?�  AK�B�B��6ff@�  @�=qB33B���                                    Bxy��  �          @�ff�Fff@�\?޸RAK�B�\)�Fff@���@���B�B�                                    Bxy�`  �          @����s33@�@Q�A��B���s33@��@�(�B��B���                                    Bxy�  �          @��R�fff@���@G
=A���B��fff@��@��B(C@                                     Bxyͬ  �          @��R�vff@���@2�\A�  B����vff@�G�@��
B33CJ=                                    Bxy�R  �          @�\��G�@�G�@;�A�Q�B�����G�@��@��B��C�
                                    Bxy��  �          @�=q���@��@HQ�AӅC+����@P��@�\)B(�CB�                                    Bxy��  �          @�����@1G�@�(�B�\C�3����?c�
@�33B>�
C(��                                    BxyD  �          @�=q�>{@��@1�A�{B�=�>{@|(�@�z�B,�HB�
=                                    Bxy�  �          @���
�H@��?�=qAuG�B��H�
�H@�(�@�33B�B�q                                    Bxy%�            @�33��
=>�  @��
BCQ�C0���
=��p�@�=qB4��CI#�                                    Bxy46  �          @�
=��\)?���@��B7z�Cٚ��\)�k�@��BG�C6�                                    BxyB�  �          @�\)�tz�@
=@�  BE�HC^��tz�>#�
@�  Baz�C1��                                    BxyQ�  �          @�  ���@�H@�  B6�CL����>��R@�G�BR{C/ٚ                                    Bxy`(  �          @��
��@.�R@�=qB6  C�)��?\)@�\)BVC,c�                                    Bxyn�  �          @���?��H@��RB1�CG�����G�@��\BC  C5^�                                    Bxy}t  �          @�\)��33?�
=@��B?C�H��33��ff@��BLQ�C9�                                    Bxy�  �          @��H��z�?�z�@�=qBB=qCB���z��@�G�BM�RC:��                                    Bxy��  �          @�=q��33@�@�z�BJ{C����33�8Q�@���B_G�C6�                                     Bxy�f  �          @���'
=@�
=@��B+�RB��=�'
=@\)@�p�Bsz�CL�                                    Bxy�  �          @�G��G�@���@���B.33B����G�@(�@ϮB{=qC
=                                    BxyƲ  �          @�G��'�@��@��B3=qB�'�@{@ָRB{�C�R                                    Bxy�X  T          @��<(�@�Q�@�=qB@�B���<(�?�G�@�{B~��CǮ                                    Bxy��  T          @�=q�<(�@��@��B:��B���<(�?�33@�
=By�HC��                                    Bxy�  �          @����@�Q�@��\BI�
B��
���?�  @�ffB��CxR                                    BxyJ  �          @�z��33@��H@ƸRBR�\B�\)��33?�ff@�p�B�\)C�
                                    Bxy�  �          A�   @��@��HBMp�B�z��   ?��
@��B�p�C
B�                                    Bxy�  �          A���(�@��@�ffB=Q�B���(�@
�H@��B�k�C
xR                                    Bxy-<  �          @�{�0��@���@��RB.
=B�k��0��@�R@�Q�Bxp�C
!H                                    Bxy;�  �          @�33���@�Q�@�33B��B�\���@O\)@��Bg\)B�{                                    BxyJ�  �          @��ÿ�@���@�
=B�B�33��@S�
@ə�Blp�B�\)                                    BxyY.  �          @���J�H@��@��RB33B��H�J�H@-p�@ə�Bb{C�                                     Bxyg�  �          @�33�7�@���@�p�B!��B� �7�@)��@�Q�Bl=qC	J=                                    Bxyvz  �          @����Fff@���@p  A�p�B�z��Fff@l(�@�{BF��C                                    Bxy�   �          @�  �8Q�@Ǯ@=p�A��\B噚�8Q�@�G�@�ffB0\)B��)                                    Bxy��  �          @���(��@�Q�?�At��B�.�(��@�ff@��B=qB��H                                    Bxy�l  T          @��H�=q@�?�G�AX(�B�\)�=q@���@�z�BQ�B�B�                                    Bxy�  �          @�p���=q@�R�����  B��ÿ�=q@�G�?��Al  B�p�                                    Bxy��  �          @�{����@У�?(��@��B�8R����@�ff@N�RA�RB�p�                                    Bxy�^  �          @�ff�N{@�33@X��A��
B�33�N{@E�@��BB�CW
                                    Bxy�  �          @���/\)?�
=@���B}Q�Cu��/\)��@�ffB��3CK(�                                    Bxy�  �          @����@��@{@���Bd��C���@�׾��R@�ffB��C9�                                    Bxy�P  �          @�z��K�@\)@�{B^C޸�K����@ÅBy��C8�H                                    Bxy�  �          @ڏ\�^{@��@�=qBMz�C�
�^{=��
@�33Bm33C2��                                    Bxy�  �          @�  �^�R@Vff@�B0z�C��^�R?�z�@�z�Bc��C!�{                                    Bxy&B  �          @�\)�XQ�@aG�@�G�B+�RCٚ�XQ�?�\)@��\Bc=qC�                                    Bxy4�  �          @��
�Q�@Mp�@�\)B1G�C�f�Q�?�{@�z�BeG�C!\)                                    BxyC�  �          @�Q��s33@N{@��B�C�R�s33?��\@�=qBN�RC!�{                                    BxyR4  �          @�  �2�\@=q@���B\C{�2�\=#�
@��\B��fC30�                                    Bxy`�  �          @��
�n�R@,(�@��B7Q�C8R�n�R?�@�p�B\��C,
                                    Bxyo�  �          @�
=�u�@i��@i��B
Q�Ck��u�?�=q@�=qBBp�Ck�                                    Bxy~&  �          @��H�R�\@ff@�G�BG=qCs3�R�\>8Q�@��\Bi{C0ٚ                                    Bxy��  �          @�\)�X��?��@�
=BP�C�X�þ���@���Be{C9�                                    Bxy�r  �          @љ��mp�@:=q@�
=B.  C���mp�?O\)@���BX��C'�                                    Bxy�  �          @љ��n{@W�@�
=B Q�C	�)�n{?�=q@�\)BS�
C ^�                                    Bxy��  �          @�  ��Q�@���@6ffAď\C�R��Q�@K�@�=qB'(�C�
                                    Bxy�d  �          @����=q@�  ?�AuC &f��=q@�
=@���BC�R                                    Bxy�
  �          @�33��\)@�?�(�Ax��B�W
��\)@�33@�z�B	G�C&f                                    Bxy�  �          @���\)@��?�Q�A(�B����\)@�ff@dz�A�C��                                    Bxy�V  �          @�p���{@�z�?}p�@�B�����{@�p�@\��A�\)C�                                    Bxy�  �          @�{�u@�  @P��A�{B�z��u@w�@��B1��C                                    Bxy�  �          @�����\)@�(�@
=A}p�B�(���\)@�ff@�\)B=qC                                      BxyH  �          @������@˅?�
=Ad��B�.���@�\)@���B	�C@                                     Bxy-�  �          @������@��
?(��@�z�B�����@�\)@W�A��HB���                                    Bxy<�  �          @������\@�(�>#�
?�B�B����\@�ff@:�HA�  B��H                                    BxyK:  �          @�����
=@љ�>�?n{B�����
=@�z�@7
=A�(�B�8R                                    BxyY�  �          @��H����@�  ���g
=B��3����@��
@(�A�(�B�\                                    Bxyh�  �          @�ff��Q�@�������C	{��Q�@��\=�?uC�)                                    Bxyw,  �          @�p����@�Q����Tz�C�
���@��?�R@�  C�{                                    Bxy��  �          @��\��{@�Q�@;�A��RB�\��{@\)@��\B%33Cc�                                    Bxy�x  �          A���R@��\@n�RAۙ�B������R@mp�@��HB8C
�)                                    Bxy�  �          A�
���H@�(�@7�A�p�B�����H@��@���B =qC�{                                    Bxy��  �          A�H���\@�@H��A�
=B�  ���\@���@��RB'�
Cff                                    Bxy�j  �          A��Q�@�
=@J�HA���C� ��Q�@u@���B#�C�                                    Bxy�  �          AG���\)@���@>�RA��\C�H��\)@a�@�ffBffC�R                                    Bxyݶ  �          A Q���z�@���@R�\A��C\)��z�@n{@�(�B(�\CB�                                    Bxy�\  �          @�{����@�z�@-p�A��\Cc�����@}p�@�33B��C�                                    Bxy�  �          @�=q���R@�@qG�A�G�C�f���R@(Q�@��B/��C\                                    Bxy	�  �          @������@�{@}p�A���C������@ff@��
B0  C�)                                    BxyN  �          @�ff���R@L(�@�z�B�
C�����R?fff@���B9�RC*.                                    Bxy&�  �          @�z����@7�@��B�RC�3���?�\@��
B={C.T{                                    Bxy5�  �          @����ff@AG�@�Q�B#\)C����ff?z�@�=qBEC-L�                                    BxyD@  �          @�{����@!G�@�B1�C������=L��@�Q�BL=qC3\)                                    BxyR�  �          @����R@P��@�
=B'��C�\���R?O\)@�z�BQ��C)�3                                    Bxya�  �          @�\)��Q�@\(�@�p�B'C{��Q�?}p�@�BV33C&�                                    Bxyp2  �          A ���.{@�ff?�=qA(�B�.�.{@���@��B
=B�Ǯ                                    Bxy~�  �          A ���1�@陚?��
AMBݳ3�1�@�=q@�Q�BB��                                    Bxy�~  T          @�{�4z�@�=q?s33@�(�B���4z�@�{@���A�z�B��                                    Bxy�$  �          @��H�o\)@�=q?��RAk�B�aH�o\)@��@�(�B\)B�                                      Bxy��  �          @������@���@�A�(�B�������@���@��\B�\CE                                    Bxy�p  �          @�z����\@�{@+�A���B��q���\@�p�@�33B!G�C�R                                    Bxy�  �          @�z���(�@�33@1G�A��RB�B���(�@�G�@�z�B#33C�                                    Bxyּ  �          @�\)���@��R@=p�A��HB�p����@��\@�Q�B%�C��                                    Bxy�b  �          A �����\@�Q�@�  A���C
���\@?\)@�z�B<z�C�f                                    Bxy�  �          AG����@�(�@��B{C+����@/\)@ÅBE(�C�
                                    Bxy	�  �          A����@�@���BC������?��@θRBT
=CaH                                    Bxy	T  �          A�R���\@��@2�\A��C���\@c�
@��\B33C(�                                    Bxy	�  �          A���x��@��ÿ^�R��G�B�G��x��@�@��A��\B�                                    Bxy	.�  �          Az����@�\=�G�?@  B��)���@�=q@L��A��\B�Ǯ                                    Bxy	=F  �          A  ��(�@�ff>Ǯ@,(�B�p���(�@\@Y��A�G�B��{                                    Bxy	K�  �          A����@�G�?�  @�Q�B��=���@�(�@�  A���B��H                                    Bxy	Z�  �          AQ����@��.{��33B��H���@ə�@9��A�z�B�                                      Bxy	i8  �          A����Q�@޸R�#�
����B��)��Q�@ȣ�@A�A�\)B�u�                                    Bxy	w�  T          AG���Q�@߮�L�Ϳ���B����Q�@˅@:=qA���B��3                                    Bxy	��  �          A���{@׮��G��G�B�����{@�33@8Q�A�(�B�#�                                    Bxy	�*  �          A33��\)@��>.{?�
=B��3��\)@���@FffA�=qC ��                                    Bxy	��  �          A�
���\@�녽u��(�B��)���\@�z�@=p�A�33B��                                    Bxy	�v  
�          A  ���
@�(�?z�H@��
C �q���
@���@mp�A�=qC)                                    Bxy	�  �          A�
��ff@�z�?�G�A)G�B�z���ff@��@�{A��C^�                                    Bxy	��  �          Aff��(�@��H?\A,Q�B�.��(�@�  @�A��CL�                                    Bxy	�h  �          Ap���G�@���@33A�33C�q��G�@�G�@��\BG�CQ�                                    Bxy	�  �          @��R���@��H@P  A��C����@c�
@���B+Q�CY�                                    Bxy	��  �          @��r�\@��@#33A�
=B�8R�r�\@�33@�z�B#
=C�                                     Bxy

Z  T          A���  @��H?�p�Ad��B�����  @�G�@�33B�C.                                    Bxy
   T          Ap�����@У�@ ��Ag�B��=����@�@�
=Bp�C�                                    Bxy
'�  �          Ap��s�
@�  @9��A�Q�B����s�
@�Q�@�  B+�
C0�                                    Bxy
6L  �          A ���Z�H@��@I��A���B�{�Z�H@�ff@�Q�B6��B��                                    Bxy
D�  �          A ���u�@�\)@(�A}G�B�=q�u�@���@�\)B
=B��=                                    Bxy
S�  �          A ���{�@�p�@
=Atz�B�  �{�@�Q�@�z�B
=C �                                    Bxy
b>  �          A ����=q@�
=?ٙ�AC�B�p���=q@��@�G�B	��B�                                    Bxy
p�  �          A�����@���?�\)AV=qB�Q�����@�ff@�\)B��B�Q�                                    Bxy
�  T          @�(��B�\@׮@$z�A�p�B�z��B�\@��@��HB+�HB�                                    Bxy
�0  �          @����L��@ᙚ?�
=A'�B��
�L��@���@��RB
  B��                                    Bxy
��  �          A   �N{@�Q�?h��@��B����N{@��@�33A���B��                                    Bxy
�|  T          @�
=�c33@��H?:�H@�\)B�33�c33@��@vffA�B�G�                                    Bxy
�"  �          @�{�mp�@߮?&ff@�p�B�  �mp�@�@o\)A���B�{                                    Bxy
��  T          @�\)�p  @�׾�z���
B�8R�p  @�z�@;�A���B��
                                    Bxy
�n  
�          @�
=�vff@�
=���
�z�B��)�vff@�  @EA���B�Q�                                    Bxy
�  �          @��mp�@�  =#�
>��RB�Ǯ�mp�@�
=@N{A��B�{                                    Bxy
��  �          @�p��Z=q@��
>�Q�@%�B�.�Z=q@�@c�
A�  B���                                    Bxy`  �          A ���\��@��=#�
>�z�B�3�\��@θR@VffAĸRB�.                                    Bxy  �          A Q��e�@�>.{?�(�B����e�@��@[�A�=qB��                                    Bxy �  �          A Q��]p�@�ff>�@QG�B�Q��]p�@�ff@l(�A��B�Q�                                    Bxy/R  �          A z��5�@�Q�?��HAE�Bހ �5�@�@��HB\)B��f                                    Bxy=�  �          @����@�R?�=qATz�B�33��@���@�G�B(�Bۅ                                    BxyL�  �          @�
=�33@�R?�p�A+33B�=q�33@��R@�\)B{B�.                                    Bxy[D  �          @�
=��\)@�{?���Ac�
B�33��\)@�
=@�z�B"{B�33                                    Bxyi�  �          A (����\@�ff@q�A�ffB����\@�{@љ�B\B�                                      Bxyx�  �          @�
=��\)@�=q?޸RAK33B�{��\)@�p�@���BG�BҀ                                     Bxy�6  �          @�ff���
@�=q?�AR�HB�����
@�z�@��HB��B�{                                    Bxy��  
�          @�
=���@�\@ ��Aj{BȸR���@���@���B%��B��)                                    Bxy��  �          @�zῙ��@�G�?�
=AbffB�
=����@���@�B$�RB�L�                                    Bxy�(  �          @��R��  @�(�@,��A��RB�(���  @���@���B;=qB�p�                                    Bxy��  �          A Q��=q@�\@333A�B�LͿ�=q@��R@��
B<��Bճ3                                    Bxy�t  �          Aff���@�Q�@   Ad(�B�W
���@��R@��B%
=B�Q�                                    Bxy�  �          A=q��{@���?�{A6�RB�W
��{@�p�@�G�B33B�G�                                    Bxy��  �          A �ÿ���@�(�@�
AmG�BɅ����@��@��B'G�B�                                    Bxy�f  T          A   �n{@�G�?�z�A"�RB��{�n{@�  @��BG�B��H                                    Bxy  �          A �þ8Q�@�ff?��\AG�B����8Q�@�ff@�=qB�B�G�                                    Bxy�  �          A\)�ǮA=q?p��@ҏ\B�p��Ǯ@أ�@��
B�B��=                                    Bxy(X  T          Aff�B�\A�?���@�Q�B��\�B�\@�33@��HB�HB��                                    Bxy6�  �          A��>k�A
=?��A�
B��f>k�@�33@��B�B�.                                    BxyE�  �          A=q?s33@�p�@
=Al��B�\)?s33@���@���B)Q�B��                                    BxyTJ  T          A ��@S33@�׿�(��H  B��3@S33@�Q�?�  AL(�B���                                    Bxyb�  �          A (�@33@�p��
�H�|  B�{@33@�\?�ffA2{B��                                    Bxyq�  �          Aff?�z�@��=p�����B�W
?�z�A   ?aG�@�\)B�u�                                    Bxy�<  �          A33?�G�@�R�G���33B��)?�G�A ��?:�H@���B�B�                                    Bxy��  �          A�\?\@����P����
=B�aH?\@��R?�@qG�B�W
                                    Bxy��  �          Aff?�G�@�  �^{��{B�Q�?�G�A z�>���@�B�33                                    Bxy�.  �          Aff?^�R@���c33��
=B�aH?^�RAG�>�=q?�
=B��q                                    Bxy��  �          A=q?�=q@�  �\(���p�B�{?�=qA (�>�Q�@#�
B��                                    Bxy�z  �          A�>8Q�@�33�Z�H�ǅB�#�>8Q�Ap�>�G�@EB�ff                                    Bxy�   �          A{?s33@���U��¸RB�8R?s33A Q�?   @aG�B��                                    Bxy��  �          A@�
@���3�
��\)B�ff@�
@��?p��@�ffB�8R                                    Bxy�l  �          A=q?\@�{�2�\��33B���?\@�(�?�ff@�ffB�8R                                    Bxy  
�          Aff?aG�@���5��\)B��R?aG�A (�?���@�B�p�                                    Bxy�  T          A33>.{@��Fff����B�ff>.{A�?Tz�@�=qB��{                                    Bxy!^  �          A{?c�
@����J�H��=qB�=q?c�
A Q�?5@��B�L�                                    Bxy0  �          A�\?�33@�(��HQ�����B�� ?�33@��?:�H@�p�B�{                                    Bxy>�  �          A��?��
@���mp���z�B�W
?��
@�{=u>ǮB�#�                                    BxyMP  �          A�?Ǯ@��
�`  ���B�G�?Ǯ@�p�>�z�@z�B��q                                    Bxy[�  �          Aff?�z�@�  �i����B��\?�z�@���=���?=p�B��H                                    Bxyj�  �          A�
?���@��
�S33����B��?���A ��?��@�{B���                                    BxyyB  �          A\)>���Ap��333���RB���>���@�ff@N{A�
=B�u�                                    Bxy��  �          A(�?�Q�@�
=�Tz����B��?�Q�Aff?!G�@�z�B�k�                                    Bxy��  �          A�\?u@���=p���z�B�u�?uA Q�?z�H@�z�B�\)                                    Bxy�4  �          A�>�ffAG���=q���B��3>�ff@��@+�A�ffB�k�                                    Bxy��  �          Aff>�A녾�����B��>�@陚@e�A�
=B��f                                    Bxy  �          A�\<�A z�s33�׮B�� <�@�Q�@@  A�  B�z�                                    Bxy�&  �          A�\�=p�A �;�=q��B�z�=p�@�ff@g
=A�B��R                                    Bxy��  �          A�R�L��AG��=p���p�B�LͿL��@�ff@N{A�B�=q                                    Bxy�r  �          A����z�@���?��\@�=qB���z�@˅@�(�B��B�=q                                    Bxy�  �          @���@�Q�?��RA-p�B�
=�@�(�@��B�B�\                                    Bxy�  �          @�33��z�@�p�?W
=@�p�B��H��z�@ʏ\@�p�B�B�(�                                    Bxyd  �          @��Ϳ�p�@�
=?aG�@�B�
=��p�@�33@�
=B
��B��
                                    Bxy)
  �          @��׿�@���>#�
?�33B�uÿ�@���@p  A�G�B�u�                                    Bxy7�  �          @��R��{@�G��Q����
BĽq��{@�Q�@9��A�  B�                                      BxyFV  �          @�Q�>�
=@��R�+����B��q>�
=@�\@G
=A�p�B�.                                    BxyT�  �          @�G�>�@��
��
=�)p�B�>�@��
@=qA��RB��                                    Bxyc�  �          @���?:�H@񙚿˅�=�B���?:�H@�(�@\)A�z�B��R                                    BxyrH  �          @�=q=�G�@���Q��NffB�Q�=�G�@�Q�@z�A~{B�L�                                    Bxy��  �          @����@���(��3�B�uÿ��@�  @
�HA�33B�.                                    Bxy��  �          @��s33@�=q��  �T��B½q�s33@�  @   At��B��)                                    Bxy�:  �          @�>.{@�=q�*=q��ffB�.>.{@�Q�?u@�B�W
                                    Bxy��  �          @�\)?�ff@�{�g
=��(�B���?�ff@�p��.{����B��
                                    Bxy��  �          AG�?�
=@޸R�.{��G�B�?�
=@��?z�H@�G�B���                                    Bxy�,  �          A�\?���@�Q�����`  B�Q�?���@�\)@�\Aj�RB�G�                                    Bxy��  �          A�@��@�33�Tz���33B��)@��@�=q?�@n{B�{                                    Bxy�x  �          A��@X��@أ��Z�H�\B}��@X��@�\>�z�?�(�B���                                    Bxy�  �          A�R@`  @�\)�J=q��z�B}z�@`  @�(�?�R@�{B�Ǯ                                    Bxy�  �          Ap�@k�@������ �\BjG�@k�@���u���HB~p�                                    Bxyj  �          AQ�@�G�@��
���R�Q�BY(�@�G�@�\��
=��Bq33                                    Bxy"  �          A�\@P��@��z�H��ffB�  @P��@���8Q쿞�RB�\)                                    Bxy0�  �          A	?�@���G
=���RB��R?�Az�?�ff@ᙚB�\)                                    Bxy?\  �          A��?�z�@�����l��B�\)?�z�@�
=?��RA[�B�u�                                    BxyN  �          AQ�?��@��R�N{���HB�\?��A��?p��@���B�aH                                    Bxy\�  �          A�?���@�  ��=q���
B�{?���A����
�.{B�.                                    BxykN  �          A��?�  @ڏ\��z���
=B�33?�  A���   �]p�B��\                                    Bxyy�  �          AG�?˅@�=q���� �
B��R?˅A���i��B�Ǯ                                    Bxy��  �          A��?�{@�
=��Q���\B���?�{A �ÿ(����G�B���                                    Bxy�@  �          A  ?��R@޸R�z=q���B��?��R@�����
��G�B�                                    Bxy��  �          AQ�@��@�{�s33����B��
@��@��=�\)?�\B�=q                                    Bxy��  
�          A��?�@�ff���
����B���?�A��L�Ϳ���B�                                    Bxy�2  �          A�@�@أ������\B��@�@�z�k���{B�                                    Bxy��  �          A�@
=q@����xQ����B���@
=q@�녽u��G�B�
=                                    Bxy�~  �          A33?��@��������HB�aH?��A Q���n{B��f                                    Bxy�$  �          A
=@
=@����\���ǅB��)@
=@�=q>�G�@Dz�B�aH                                    Bxy��  �          A(�@{@�{�xQ���  B��=@{@�ff<�>8Q�B���                                    Bxyp  T          A  ?�  @����
=���B�(�?�  @�{���R��B��                                    Bxy  �          A��?u@Ǯ��33�!(�B��=?uA������2=qB�p�                                    Bxy)�  �          A�?��
@������
�(��B�
=?��
A z���Tz�B��
                                    Bxy8b  �          A��?��@�Q���ff���B�(�?��A (����H� ��B��)                                    BxyG  �          A�@
�H@ə������33B�{@
�H@��R���
��B��{                                    BxyU�  �          A�@\)@�33��p��33B��R@\)@�\)���H�:�RB���                                    BxydT  �          A  @!G�@�=q��(��(�B��H@!G�@�{��Q��7�
B��R                                    Bxyr�  �          A�\@!G�@�  �������B��@!G�@�p���ff�=qB��3                                    Bxy��  �          A�H@#�
@�=q��p���\B�B�@#�
@��\��G��&{B�Ǯ                                    Bxy�F  �          A�@&ff@�Q���z��&�HB�p�@&ff@�G���
�ap�B��                                    Bxy��  �          A
�\@u@�33���
��Bc  @u@��Ǯ�&{B}
=                                    Bxy��  �          AQ�@i��@�  ���
�z�Ba�
@i��@�{�����T(�Bz�                                    Bxy�8  y          A(�@qG�@�ff���\�Q�B]p�@qG�@��
��Q��S�B{                                    Bxy��  �          A�\@s33@�Q���=q���B]��@s33@陚��Q��9G�By��                                    Bxyل  �          A(�?�Q�@�33�ʏ\�CG�B���?�Q�@�G��7
=���HB��                                    Bxy�*  �          A�\?��@�����p��FG�B��?��@����<����(�B�ff                                    Bxy��  �          A�\?h��@�����(��Ep�B��)?h��@�33�7����B�Ǯ                                    Bxyv  �          A?�z�@������H�F33B�Q�?�z�@�\)�9�����\B�                                      Bxy  �          A�?�Q�@�33�θR�K�HB��?�Q�@�(��E��(�B�=q                                    Bxy"�  �          A=q���HA=q@'
=A�
=B˳3���H@���@ə�B9
=B��                                    Bxy1h  �          AG��E�A�
���E�B��E�@���@��A�RB�p�                                    Bxy@  �          Ap��(�Az�   �Q�B���(�@�(�@z�HA�33B��f                                    BxyN�  �          A	�ǮA�H������RB�W
�Ǯ@�z�@L(�A�(�B��3                                    Bxy]Z  �          Az�>\A
=��\)�333B��3>\@�p�@*=qA�ffB��                                     Bxyl   �          A\)���A���aG���Q�B��\���@�{@X��A�B�33                                    Bxyz�  �          A��>�33@��������=qB���>�33@��R?�ffAL  B��f                                    Bxy�L  �          Aff@
=q@�
=��\)��\B���@
=qAQ쿥��p�B�\)                                    Bxy��  �          A��?ٙ�@��������ffB��?ٙ�A���g�B�B�                                    Bxy��  �          A��?�33@�ff��p���HB��?�33A�\�\)�a�B��                                    Bxy�>  �          A��?   A��h������B�=q?   A�\?��@��
B��q                                    Bxy��  �          A  =�\)A���
�g
=B��=�\)A�H@{Aw
=B��                                    BxyҊ  �          A33?=p�A	p����X��B�#�?=p�A�@�RA�  B�                                    Bxy�0  �          Aff���@�ff@3�
A�33B�p����@��@���B)p�C�R                                    Bxy��  �          A����@���@�  Bz�C�R���?n{@�ffBKz�C+33                                    Bxy�|  �          A�H����@�p�@��BQ�C������?�33@�BHQ�C&�q                                    Bxy"  �          A=q�Å@��\@�p�B=qC���Å?���@�\BE��C'��                                    Bxy�  �          A{�Å@�Q�@�  B=qC���Å?���@�Q�BC��C%^�                                    Bxy*n  �          A���@�(�@�B	z�C���?�p�@�Q�BDG�C${                                    Bxy9  T          A�\�\@���@�ffB	�\CǮ�\?�p�@陚BDQ�C$!H                                    BxyG�  T          A�����H@���@��\BCQ����H?��H@�z�BA��C$J=                                    BxyV`  �          A�\����@�
=@�\)B
��C������?���@�\)B9�C*33                                    Bxye  �          A����33@�z�@�Q�B�C^���33?k�@�B<�C,\                                    Bxys�  �          A  ���
@���@�p�B 33C	�����
@p�@�  B@p�C �                                    Bxy�R  �          A�\��ff@\@��
A�(�C&f��ff@5�@�Q�BC
=C��                                    Bxy��  �          A(���p�@���@��A��RCu���p�@(�@ٙ�BCp�C�q                                    Bxy��  �          A����(�@ᙚ@b�\A��HB���(�@��\@�33B;��CW
                                    Bxy�D  �          A�\��z�@�@eA��
B�q��z�@�@ۅB;�RC(�                                    Bxy��  �          A�����@�z�@`  A��HB������@�p�@�33B533C	n                                    Bxyː  �          A������@��@w�A˙�C �����@fff@��B;  CY�                                    Bxy�6  �          Aff��G�@��R@|��A��HC����G�@J�H@���B8Q�C
=                                    Bxy��  �          A
=��{@�
=@���A���C p���{@W
=@�33B?ffC�                                     Bxy��  �          Az����@�z�@�p�A���C����@Mp�@�{B?�
Cٚ                                    Bxy(  �          Az���z�@Ǯ@S�
A�B�.��z�@n�R@�  B8�C�
                                    Bxy�  �          A
=�H��@�?\A!B�.�H��@��H@���B�
B�                                    Bxy#t  �          A����A\)>8Q�?���B��
����@�{@���A�  B�L�                                    Bxy2  �          A\)�*=q@��R?�  @�G�B�  �*=q@�(�@��
B��B�Q�                                    Bxy@�  �          A��5�A	?J=q@�Q�B�p��5�@�G�@�Q�B��B��
                                    BxyOf  �          A��A�A33>��?�ffB�ff�A�@�@�
=A��
B�p�                                    Bxy^  �          A��hQ�A(�=�?:�HB���hQ�@�\)@�z�A�Q�B�aH                                    Bxyl�  �          A!���RA녿(��^�RB�\���RA
=@��A�z�B�k�                                    Bxy{X  |          A/
=��p�A=q�#�
�aG�B����p�A{@�{Aϙ�B��f                                    Bxy��  �          A,��?��?�A%G�B��\BXQ�?���h��Ap�B�{C��                                    Bxy��  �          A*�\?���@�G�AffBy�B��q?��ÿ
=A(z�B��3C��H                                    Bxy�J  �          A�H@*=q@���@ᙚB5�HB���@*=q?�p�A��B�ǮB\)                                    Bxy��  �          A��E�@�  @�(�B>p�B®�E�?�A  B�B�B�                                     BxyĖ  �          A��ff@���@��B  B��ff@h��@�ffBrp�B��R                                    Bxy�<  �          A�
�l(�@��R@��HBG�B� �l(�@�@��\Bs�\C��                                    Bxy��  �          A����z�@��\@�G�B\)C��z�?�
=@�Q�B]=qC^�                                    Bxy��  �          AQ���{@���@�p�B�CY���{?�G�@�Q�BT�
C k�                                    Bxy�.  �          A������@��H@�G�B,�C	������?0��@��Bc�
C+��                                    Bxy�  �          A�����@��
@ÅB/z�C�{���?0��@�
=Bh�C+��                                    Bxyz  �          A�����@���@�\)B+{C
�R���?(��@�=qB`��C,8R                                    Bxy+   �          A\)��ff@�33@��RB{C^���ff?���@�{BI�C(O\                                    Bxy9�  �          A
�\��(�@�=q@�  B\)C����(�?�@�(�BH  C%:�                                    BxyHl  �          A	����z�@�@�\)A�CQ���z�?�\@�\)B:{C"��                                    BxyW  �          A
�R�\@k�@�
=B  C�
�\?(��@���B4��C-�\                                    Bxye�  T          A
�H���
@9��@��B�\C����
���
@��
B*C4��                                    Bxyt^  �          A�R��ff@(��@�  Bp�Cz���ff��p�@��
B$�C733                                    Bxy�  T          A���ڏ\@%�@��B
=CY��ڏ\��@�p�B#ffC8\                                    Bxy��  �          Az�����@(�@�G�B��C ����Ϳ�@���B�HC8�q                                    Bxy�P  T          A���33@�R@�33BG�C!�q��33�G�@��B
=C:}q                                    Bxy��  �          Aff��?�@��B{C$������p�@�\)B ��C>k�                                    Bxy��  �          A(���Q�?�@��RB"Q�C'����Q��z�@�z�B   CBW
                                    Bxy�B  �          A  ��33?�(�@��B�\C)�\��33���@�ffBQ�CC(�                                    Bxy��  �          AG���{@   @�G�B��C#�f��{�Q�@��B��C:�q                                    Bxy�  �          A����Q�?���@��BffC))��Q쿵@��HBC?xR                                    Bxy�4  �          Az���?�33@�Q�BG�C*������@���B��C@5�                                    Bxy�  �          A(���p�?�p�@��A�Q�C$����p��!G�@�\)B
��C9                                    Bxy�  �          A����
@&ff@�  A�C����
�#�
@�ffB��C4J=                                    Bxy$&  z          AG���z�@%@��A�\C {��z�#�
@�B�
C4O\                                    Bxy2�  .          A����@>�R@�G�A�
=C�����>���@��RBQ�C0�)                                    BxyAr  
�          A���33?�=q@��B  C%�=��33�aG�@��
B33C;\                                    BxyP  T          A���=q?�
=@���B�
C&����=q�z�H@��B�HC;�f                                    Bxy^�  
�          A
=��{?\@�
=B�HC'����{����@��B�C=ٚ                                    Bxymd  �          A���ƸR�\@�(�B%ffCA��ƸR���@��\A�
=CU�{                                    Bxy|
  �          A	��{�s33@�B��C<{��{�\(�@\)A�\)CO:�                                    Bxy��  �          A��ᙚ@3�
@�{A�G�CE�ᙚ>���@�G�B
=C1�\                                    Bxy�V  �          A\)����@�H@��A��HC �R���þ8Q�@�33BC5z�                                    Bxy��  �          A\)��Q�@=q@���A�{C!���Q�aG�@�z�B�C5�{                                    Bxy��  �          A\)��z�?L��@��BG�C-c���z��(�@�B	��CC�                                    Bxy�H  �          A\)��G�?�(�@�\)B��C*+���G���(�@�p�B	\)C?��                                    Bxy��  �          A33��(�?�=q@�G�B�HC)n��(����
@��BG�C>0�                                    Bxy�  "          A����
?˅@���B�C'h����
��ff@�B	�\C<c�                                    Bxy�:  �          A(���{@G�@��A�
=C$O\��{��@�ffB	�HC8�                                     Bxy��  �          A(���ff?�@���A�C%����ff�8Q�@�33B\)C9�R                                    Bxy�  �          A(���
=?�@���A��C%&f��
=�#�
@�(�B��C9{                                    Bxy,  �          A  ��R@:=q@{�A���C
=��R?�\@��
B��C/�                                    Bxy+�  �          A���
=@z=q@[�A��C���
=?��@�{B
\)C&Ǯ                                    Bxy:x  �          A  ��\)@˅?��RA]C ���\)@�\)@�{B��C
�                                    BxyI  �          A��N{@�?�ffAz�B�=q�N{@��H@�ffB33B��q                                    BxyW�  �          A���\)@��@p�Aw
=C����\)@�33@�\)BG�C��                                    Bxyfj  �          A	���\)@]p�@���A��
C����\)?s33@�
=B{C,�                                    Bxyu  �          A	p���@�(�@#33A�z�C����@c�
@���Bz�C�                                    Bxy��  �          A	G���  @�
=@+�A�=qC	�f��  @W
=@��\B�CǮ                                    Bxy�\  �          A������@�33@�\Ay�C	�����@j�H@�=qB	=qC�                                    Bxy�  �          A  ����@�\)?ٙ�A8Q�C�����@���@��A���CaH                                    Bxy��  �          A�����@�G�?��HA�
C�\����@��H@��A�Q�C��                                    Bxy�N  "          A�R��=q@�ff?��A\)CY���=q@��R@�z�A��
C}q                                    Bxy��  �          A��{@�  ?�A�C�f��{@��
@{�A�Q�C��                                    Bxyۚ  �          A�R��(�@��?xQ�@�(�C	G���(�@�z�@mp�A�G�Cff                                    Bxy�@  �          A�R����@�=q?.{@��
C������@�z�@j=qA�Q�C�=                                    Bxy��  �          A�\��@ʏ\?��\@�Q�C����@�{@��HA�
=C	��                                    Bxy�  �          A{��33@��
?L��@���C  ��33@��\@z�HA��
Cn                                    Bxy2  �          A�����@��?s33@�p�C �����@�@�z�A陚C��                                    Bxy$�  �          AQ���{@��ý#�
��z�B���{@У�@q�A̸RB��                                     Bxy3~  �          A����R@�������@��B�W
���R@�(�?�
=AMp�B���                                    BxyB$  �          AQ���
=@�\)�(����B�z���
=@�
=?ǮA$��B��H                                    BxyP�  �          A�R�n�R@���{�q��B�{�n�R@�?��AB�\B�\)                                    Bxy_p  �          A\)���@�Q�>�{@
�HC�����@��@mp�A���C�                                    Bxyn  �          Az����
@�
=���H�FffB�B����
@�\)@O\)A�=qC��                                    Bxy|�  �          A�
��33@�{��  �˅B�Q���33@ʏ\@Z�HA���C:�                                    Bxy�b  �          A���  @�  =��
?
=qB�����  @ƸR@o\)A�C5�                                    Bxy�  �          A�����@���>���@%B������@��R@}p�A�p�C��                                    Bxy��  �          A���33@�
=?8Q�@�33C ���33@���@���A�Q�C�q                                    Bxy�T  �          A���ff@ۅ?z�H@���C����ff@�p�@�=qA�{Cn                                    Bxy��  �          A  ��z�@�{?J=q@�G�C\��z�@��H@�{A޸RC:�                                    BxyԠ  �          A�
��G�@��?\(�@�
=C J=��G�@��
@�G�A�(�C�
                                    Bxy�F  �          A�����@�=q?�{@ᙚCc����@��\@��A�C	xR                                    Bxy��  �          Ap���z�@�=q?Tz�@�\)C �\��z�@�@�G�A�C�                                    Bxy �  �          A��z�@߮?�  A\)C ����z�@���@�=qA��C�                                    Bxy8  �          A���@ҏ\?�
=@�RC8R��@��\@�33A�{C�
                                    Bxy�  �          AQ����@�  @�
AS�C)���@5@���A��CL�                                    Bxy,�  T          AQ���=q@���@
�HA^�RC���=q@B�\@�G�A㙚Cn                                    Bxy;*  T          AG���  @�
=@\)AdQ�C�\��  @J=q@�{A�z�Cp�                                    BxyI�  �          A���@�G�@�A]��CW
��@]p�@��A�C�                                    BxyXv  �          A=q���@�{@1G�A���CQ����@)��@�Q�A�C 
                                    Bxyg  �          A����
@�z�?���Az�C
\)���
@�33@�
=A�z�C��                                    Bxyu�  �          A
=��=q@�{@Q�A��
C��=q@
�H@��\Bp�C#s3                                    Bxy�h  �          A���  @���@N�RA�{CT{��  @�
@�{BC$�f                                    Bxy�  �          A(���Q�@�p�@]p�A��C�R��Q�?���@��HBp�C&(�                                    Bxy��  �          A����z�@��@~�RA�p�C�3��z�?��@�  B\)C(=q                                    Bxy�Z  �          A=q���@�33=#�
>�\)B��Ϳ��@�G�@w�A���B�\                                    Bxy�   �          A��@\)Az��1G����
B��=@\)A��?��A5B��                                    Bxyͦ  �          A��?�ffA
�H��
=�+�B���?�ffAp�@;�A�\)B��                                    Bxy�L  �          A��?�\A�
��ff���B��=?�\A��@g�A�z�B�                                      Bxy��  �          A<�A��������B��{<�A�H@W
=A�B��\                                    Bxy��  �          A��=uA33�aG����
B�=q=uA�R@p  A��B�(�                                    Bxy>  �          A(��=p�A�R�p������B�p��=p�A�H@k�A���B�L�                                    Bxy�  �          A��>�
=A��������B�G�>�
=A��@	��Aa��B�W
                                    Bxy%�  �          A33���AzῪ=q�	B��
���A(�@QG�A�ffB�W
                                    Bxy40  �          A�׿�  AG��0����z�B�k���  @�\)@vffA��
B�                                    BxyB�  �          A�����A
==���?!G�B��f���@�R@��RA��B׳3                                    BxyQ|  
f          A��=qA  ?�
=A�Bә��=q@�(�@���B(�B��                                    Bxy`"  
�          A���8Q�A�?�G�A��Bڨ��8Q�@�33@�{B��B��)                                    Bxyn�  �          A\)�xQ�@У�@��RA�\B� �xQ�@`��@�p�BS��C	�\                                    Bxy}n  
�          A	���=q@��
@��
B=qB��H��=q@=q@�G�B`G�Cff                                    Bxy�  
�          A�
��ff@��@��\A�z�B��{��ff@1�@�(�BNz�C�3                                    Bxy��  �          AQ���=q@���@��HA�\B�
=��=q@K�@ҏ\BK=qC��                                    Bxy�`  
�          A�
�\)@�Q�@��A�B�  �\)@Vff@�ffBPz�C��                                    Bxy�  �          A���p�@Ǯ@y��AۅB��=��p�@\(�@�Q�BI
=C�                                     BxyƬ  �          A33��Q�@�  @aG�A�{B�k���Q�@vff@��BB\)C.                                    Bxy�R  "          Aff��z�@���@�A|(�B�����z�@�
=@��
BG�C��                                    Bxy��            A��z�@���>�=q?�=qC 5���z�@���@`  A�p�C��                                    Bxy�  
�          AG����@���?�(�A�
C	�f���@�p�@u�A�  C}q                                    BxyD  
�          A����Q�@��?Q�@��Ch���Q�@Tz�@7�A�=qC��                                    Bxy�  z          A����33@�=q?�@hQ�C���33@^�R@'�A��RC�                                    Bxy�  
�          A����{@��=L��>���CE��{@a�@ffAj�HCٚ                                    Bxy-6            A  �љ�@��?k�@�z�C&f�љ�@r�\@P  A�  C�3                                    Bxy;�  
�          A33����@�33?}p�@޸RCff����@l��@R�\A�33Cu�                                    BxyJ�  �          A�����
@���=�G�?J=qC	L����
@��\@;�A�z�C��                                    BxyY(  �          A���p�@��H�aG���ffCh���p�@�p�@.�RA�
=C
�f                                    Bxyg�  
�          A���ə�@��
�#�
��z�C�
�ə�@�p�@)��A�33Cz�                                    Bxyvt  T          AQ��˅@���>�=q?�\)C^��˅@�@7�A�ffC(�                                    Bxy�  �          A(��θR@�z�=���?.{C���θR@���@)��A��
C��                                    