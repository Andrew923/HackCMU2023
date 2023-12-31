CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230129000000_e20230129235959_p20230130021546_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-30T02:15:46.917Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-29T00:00:00.000Z   time_coverage_end         2023-01-29T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx_�-�  
�          @�  ��\)@dzῂ�\�n{B�
=��\)@j�H?z�A�B�                                    Bx_�<&  
Z          @��þ��H@a�?��
A�{Bî���H@p�@EBL  B�aH                                    Bx_�J�  �          @\)���
@>{@$z�B#��B������
?�{@e�B�k�B�#�                                    Bx_�Yr  "          @~{��(�@\��?��A�z�B�\��(�@@J�HBT\)Bȣ�                                    Bx_�h  �          @��>\)@J=q@$z�B33B�>\)?��@j=qB(�B�Ǯ                                    Bx_�v�  
�          @���>�@dz�?�A�  B�p�>�@�R@H��BM  B�.                                    Bx_��d  
Z          @�33?&ff@_\)?�(�A�\B���?&ff@@P��BU  B�\                                    Bx_��
  �          @\)>�
=@7
=@(��B)�B�=q>�
=?��R@fffB�ffB�u�                                    Bx_���  T          @y��>�{@%@8Q�B?�B�>�{?�33@l��B�(�B��                                    Bx_��V  
�          @�  ?5@�R@C�
BG(�B��?5?z�H@s�
B��HBW��                                    Bx_���  
�          @��>L��@8Q�@7�B3Q�B��f>L��?�z�@s�
B��HB���                                    Bx_�΢  �          @���<#�
@@  @-p�B(�B��{<#�
?˅@n{B���B�33                                    Bx_��H  �          @���33@R�\@#�
B��B�\��33?�@l��Bx��BȀ                                     Bx_���  
�          @�  �(�@e�@��A�\)B�uÿ(�@�@_\)B]ffB�u�                                    Bx_���  �          @z�H��(�@QG�@��B{B���(�@�@S�
Be��B�W
                                    Bx_�	:  "          @c33��p�@5�������B����p�@4z�?#�
A((�B�L�                                    Bx_��  
�          @R�\���@*=q�#�
�8��B��R���@�R?uA��HB�k�                                    Bx_�&�  
Z          @\(���@.�R�����p�B�ff��@+�?5A>ffB�u�                                    Bx_�5,  
�          @W����R@�Ϳ��
��Q�C{���R@*=q=�Q�?�G�B��                                    Bx_�C�  �          @b�\��@�Ϳ�=q����C	����@#�
��z���
=C��                                    Bx_�Rx  T          @tz��7�@�:�H�333C���7�@(�>��R@�
=C��                                    Bx_�a  
�          @u�(�@E>�z�@��HB��3�(�@,(�?�ffA�z�C.                                    Bx_�o�  
�          @qG����@P  ?��HA�Bڅ���@ff@+�B7(�B��                                    Bx_�~j  
�          @s33��
=@>�R@�\B33B�  ��
=?�  @Tz�Bv
=B��)                                    Bx_��  
�          @r�\?8Q�@=q@333B@\)B��{?8Q�?��\@c33B�{B[�                                    Bx_���  
�          @`  =��
?��@O\)B��B�B�=��
���R@Z�HB�Q�C���                                    Bx_��\  
a          @c�
?}p�@�\@+�BGffB�k�?}p�?=p�@R�\B��Bp�                                    Bx_��            @r�\?�  @z�@$z�B,  Bd��?�  ?��@S�
Bt�\B\)                                    Bx_�Ǩ  �          @g�?�{@&ff@�\B{ByQ�?�{?\@;�B\�RB@Q�                                    Bx_��N  
�          @[�?�p�@Q�@G�B��BK33?�p�?���@.{BT
=B{                                    Bx_���  
�          @a�?��R@\)@=qB,��Btp�?��R?��@G�Bzp�B Q�                                    Bx_��  
Z          @\(�?aG�@)��@   BffB�W
?aG�?�=q@:�HBhBs�                                    Bx_�@  T          @B�\?�R@�?�=qB�B�k�?�R?�(�@=qBb=qB�p�                                    Bx_��            @8�ÿ�
=?J=q@z�B[ffC�H��
=�u@(�Bm\)C=s3                                    Bx_��  
�          @(Q����?�33B9\)C7�����c�
?�
=B=qCN=q                                    Bx_�.2  T          @Tz��*�H���?��A�  CQ���*�H��\)>���@޸RCW�                                    Bx_�<�  
(          @Fff�&ff��
=?h��A�=qCP�)�&ff��
=>k�@�33CT�H                                    Bx_�K~  �          @H���!G����R?�\)A��CR���!G���=q>��@�CW�                                    Bx_�Z$            @8Q��
=���
?��HA�CW�R�
=���>��A�C]�3                                    Bx_�h�  
�          @<���
�H��Q�?�{Aܣ�CU� �
�H��\)?&ffAK
=C\�                                     Bx_�wp  
�          @
=��=q��Q�>�z�@��
Cc
=��=q��z��G��,Q�Cb�                                     Bx_��  �          @0  ���Ϳ��R���
��Cb����Ϳ�ff�W
=���C`(�                                    Bx_���  �          @A��(����R?
=A3�
C^:��(���
��\)����C_&f                                    Bx_��b  T          @E��	���?+�AJ=qC`.�	���(���  ��z�Cah�                                    Bx_��            @O\)����\)?z�A%G�C`�R����녾Ǯ��\)Ca:�                                    Bx_���  
/          @R�\��H��>�AG�C^���H������C^�                                    Bx_��T  	�          @QG��-p���p�?��A(�CT�
�-p����W
=�i��CU��                                    Bx_���  "          @N�R�3�
���>\@�=qCP�q�3�
�Ǯ��\)����CQ�                                    Bx_��  T          @O\)�@  �h��?+�A@(�CD��@  ��{>k�@�p�CH@                                     Bx_��F  "          @W��'��>�
=@��
CZ���'������CZp�                                    Bx_�	�  !          @Z=q�$z��(�>k�@y��C\ff�$z���0���;�C[)                                    Bx_��  "          @Z=q�G
=��33�B�\�QG�CLG��G
=�����@  �K�CI
                                    Bx_�'8  "          @X���H�ÿ��
���
��\)CJ=q�H�ÿ�녿���"�HCG�                                    Bx_�5�  �          @[��L(���G���G���G�CI���L(���{�(��$Q�CG�                                    Bx_�D�  �          @XQ��K����׽�Q����CG���K��}p������CEY�                                    Bx_�S*  �          @Z=q�N{���þ�  ��CFff�N{�aG��+��4(�CC8R                                    Bx_�a�  
�          @Z=q�P�׿u���R���CD^��P�׿@  �+��5��C@�                                    Bx_�pv  T          @X���N{���
�����z�CE�=�N{�W
=�(���333CB�{                                    Bx_�  �          @]p��R�\���
�k��qG�CEff�R�\�Y���!G��'�CBu�                                    Bx_���  T          @`  �U���ff�aG��b�\CE���U��^�R��R�$��CB��                                    Bx_��h  �          @`���Vff���\�B�\�I��CD��Vff�Y���
=��CBJ=                                    Bx_��  N          @b�\�W�����������CEp��W��\(��+��.�HCBO\                                    Bx_���  
�          @i���Z=q���R�\���CH��Z=q�z�H�\(��Yp�CD                                    Bx_��Z  
Z          @g
=�O\)��{�Q��QCJǮ�O\)�h�ÿ�ff��
=CC�3                                    Bx_��   T          @g��J=q��(��O\)�P��CL��J=q���\�����\)CE�=                                    Bx_��  �          @h���Z=q��
=����\CG
=�Z=q�aG��k��ip�CBz�                                    Bx_��L  
�          @j�H�W����Ϳ   ��33CI�)�W����
��  �~�HCE
=                                    Bx_��  
�          @s�
�Tz��(��=p��2�\COh��Tzῢ�\��\)����CI�                                    Bx_��  �          @u��QG����ÿG��;�
CQ(��QG����Ϳ�����=qCJz�                                    Bx_� >  
Z          @n�R�N�R��33�Y���RffCO
�N�R����Q���ffCG�H                                    Bx_�.�  
�          @g
=�I����G��aG��a�CM�3�I�����
����33CF�                                    Bx_�=�  �          @`���W
=�aG��n{�vffC7���W
=>8Q�p���x��C1�                                    Bx_�L0  �          @_\)�QG��B�\�^�R�hz�CA(��QG���33�������\C:{                                    Bx_�Z�  �          @`���Dz�&ff��(���z�C?�f�Dz�    �����ܣ�C4�                                    Bx_�i|  
�          @c�
�P�׿p�׿�ff��{CD(��P�׾�G������  C;��                                    Bx_�x"  	�          @a��I����ff��z���=qCFs3�I�����H���R��  C<�                                    Bx_���  T          @^�R�5�����
=�Ù�CM8R�5�&ff���\)C@�f                                    Bx_��n  �          @`���-p��ٙ���33���HCT\�-p��}p����H��HCH)                                    Bx_��  "          @\���"�\��zῦff��(�CX���"�\��p����H�(�CM�=                                    Bx_���  
Z          @`  �C�
�n{�����ffCD�H�C�
��z��33��C9p�                                    Bx_��`  �          @`���8�ÿ������مCJ{�8�þ�(���Q��33C<�                                     Bx_��  
�          @a��.�R���H��z����CP��.�R�+��
=�Q�CA��                                    Bx_�ެ  
Z          @w��^�R>8Q��33��G�C1\�^�R?TzῸQ���  C&��                                    Bx_��R  
�          @����c�
=�G�������HC2B��c�
?Y����
=��p�C&�=                                    Bx_���  T          @w��[�?   �ٙ���\)C+�q�[�?�녿�{��G�C!��                                    Bx_�
�  �          @p  �Z�H?z�H��\)��\)C#���Z�H?��Ϳ!G��  C�                                     Bx_�D  �          @vff�b�\?z�H��33���C$��b�\?�{�(���
=C�                                    Bx_�'�  "          @n�R�XQ�>�z�����33C/�XQ�?aG��������C%\)                                    Bx_�6�  
�          @g��N{�#�
��\)���
C4Ǯ�N{?(���  ��\)C)Y�                                    Bx_�E6  "          @aG��C33����
=��\C<�=�C33>�  ��(���
=C/\)                                    Bx_�S�  �          @]p��8�ÿ=p���p���\)CBT{�8��<#�
����p�C3Ǯ                                    Bx_�b�  
�          @_\)�Dz���Ϳ�=q��\)C5��Dz�?����R��=qC*.                                    Bx_�q(  T          @y���c33>�
=��ff��p�C-G��c33?�  ��G���z�C$L�                                    Bx_��  �          @q��Tz�?:�H�˅��33C'���Tz�?�ff����  C�{                                    Bx_��t  T          @.�R�   ?��+��t��C� �   ?��ýL�Ϳ�
=C                                    Bx_��  
�          @(��aG�@\)�u�\B��H�aG�@?Tz�A��RB���                                    Bx_���  "          @1녿   @�>k�@�z�B�z�   @ ��?�{A�\BϸR                                    Bx_��f  T          @?\)=u@5?��A<��B��==u@Q�?�B33B�\                                    Bx_��  
�          @@  ?
=@0  ?G�Aw\)B��3?
=@{?�ffB=qB�#�                                    Bx_�ײ  
�          @a�?�@2�\?���A�
=BlQ�?�@33@z�B$��BJ�                                    Bx_��X  
�          @mp�?�z�@=p�?��A�=qBr��?�z�@�@��B'{BR�R                                    Bx_���  
�          @o\)?�\)@=p�?���A�G�B�=q?�\)@ ��@5BE��B^z�                                    Bx_��  �          @i��?У�@!�@�\BBdff?У�?\@7�BP�\B+��                                    Bx_�J  
�          @O\)?��H?˅@ ��BMBR�?��H>�@<(�B�{A���                                    Bx_� �  N          @S�
?Ǯ?�p�@	��B"  BO�?Ǯ?z�H@0  Ba
=B p�                                    Bx_�/�  
�          @XQ�?�z�@z�?���B�B=(�?�z�?�Q�@\)B?G�A��                                    Bx_�><  �          @O\)?��@   ?��HA�B:��?��?�@
=B;�A��R                                    Bx_�L�  
�          @HQ�?�?�ff?��BQ�B3ff?�?u@ffBC�A�
=                                    Bx_�[�  
`          @L��?��
?�=q?�
=B=qB7?��
?p��@\)BL�Aޣ�                                    Bx_�j.  
�          @Mp�?��H?��?��HB33B>�H?��H?xQ�@"�\BP�A�R                                    Bx_�x�  �          @\)?���?�(�?�Q�B��Be��?���?��
?��RBS��B+�                                    Bx_��z  �          @&ff?��H?�p�?�ffBz�B5G�?��H?B�\@   BI�A�(�                                    Bx_��   
�          @%�?���?�p�?��B�B>��?���?:�H@�BV��A��
                                    Bx_���  N          @!�?��?�  ?ǮB�HBE�
?��?G�@G�BU33A�=q                                    Bx_��l  (          @5�?��?�  ?��B�B)?��?@  @BE
=A�                                      Bx_��  T          @   ?�  ?�p�?˅B&Q�B_��?�  ?@  @�\Bi�B�H                                    Bx_�и            @?��
?�G�?�=qB�B_=q?��
?aG�?�BT�B"p�                                    Bx_��^  
�          @)��?   >��@#33B�z�B��?   �(�@ ��B�.C�Ф                                    Bx_��  T          @AG�?+�?@  @5�B�  BAG�?+�����@9��B���C�~�                                    Bx_���  �          @I��?p��?�(�@#33B^z�Be?p��>�p�@:�HB���A�(�                                    Bx_�P  "          @.�R?u?�\)?�p�B:  Bm�H?u?:�H@p�BffB                                      Bx_��  T          @0��=�Q��33?��B$�C�1�=�Q��%�?^�RA�33C��3                                    Bx_�(�  T          @3�
?n{��{@
=Bi�
C�3?n{��
=?�  B=qC���                                    Bx_�7B  �          @E?�\)?���@Q�BK\)B�?�\)=�\)@'
=Bh��@'�                                    Bx_�E�  �          @QG�?�(�?�G�@%�BQ=qA���?�(��u@1G�Bh�C��                                    Bx_�T�  N          @j�H?�@!G�?��
A�(�BS{?�?�  @Q�B*{B)Q�                                    Bx_�c4  
.          @c�
?�
=?��H@7�BX�
B?�
=<��
@G�Bv�\?+�                                    Bx_�q�  
�          @e�?�?��@8Q�BW�
A�
=?���@C�
Bm�C��                                    Bx_���  (          @c�
?�z�?�\)@,(�BC�HBT  ?�z�?333@L��B(�A�ff                                    Bx_��&  �          @g�?�G�?�33@*�HB>��BM��?�G�?=p�@L(�Bx�A��                                    Bx_���  T          @g
=?˅?�33@1G�BI�
B8ff?˅>��@L(�BzG�A��\                                    Bx_��r  �          @c�
?��\@@%�B:{Bk{?��\?u@K�B}�
B{                                    Bx_��  T          @S33?�\)?��H@��B:33Bq  ?�\)?h��@<��B��Bff                                    Bx_�ɾ  T          @XQ�?���@��@  B(�B�Q�?���?��R@<(�Bs��BD�                                    Bx_��d  T          @XQ�?�ff@
�H@G�B,�B�{?�ff?�33@:�HBwffB=�
                                    Bx_��
  �          @^�R?(�?�{@6ffB^�B���?(�?&ff@VffB�8RB:Q�                                    Bx_���  T          @k�?B�\@G�@2�\BE(�B�=q?B�\?��@\(�B���BY�                                    Bx_�V  �          @e?(��@=q@#33B6G�B�aH?(��?��\@QG�B���Bzff                                    Bx_��  
�          @>�R?��@�
@z�B0�B�� ?��?���@,��B�B�Bw��                                    Bx_�!�  (          @/\)>���?�{?�z�B4
=B���>���?�G�@{B�ffB���                                    Bx_�0H  �          @3�
?
=q?޸R@�BE�RB�G�?
=q?O\)@'�B��Ba
=                                    Bx_�>�  �          @0  ���
?n{@!�B�B����
��Q�@,(�B��C~��                                    Bx_�M�  �          @H��>�{>��@C33B�#�BJ��>�{�B�\@?\)B��C��                                    Bx_�\:  �          @P  �8Q�>�33@L��B�  B�3�8Q�^�R@FffB��HC��                                    Bx_�j�  �          @=p��Ǯ>��@:=qB��Cc׾Ǯ�s33@0  B�#�Cw�H                                    Bx_�y�  T          @P�׿��;���@AG�B�z�CCQ쿌�Ϳ�33@,(�Ba��Cg�
                                    Bx_��,  �          @p�׿h�ÿ
=@fffB�k�CU&f�h�ÿ��@G�B`ffCtO\                                    Bx_���  �          @w��J=q�.{@n�RB���C\���J=q�G�@Mp�B_Cxz�                                    Bx_��x  
�          @xQ�@  �!G�@p  B��C[�3�@  ��p�@P  Bc��Cy(�                                    Bx_��  T          @z�H����ff@vffB�� C_B�������@Z=qBs  C^�                                    Bx_���  
T          @u�=u��@q�B���C���=u����@UBu\)C���                                    Bx_��j  
�          @{��#�
�W
=@z=qB��RC�AH�#�
���@c�
B�� C�AH                                    Bx_��  T          @��׼�>u@�  B�.B�8R����G�@s�
B�z�C�S3                                    Bx_��  "          @�����
��=q@�(�B��C�o\���
��\@o\)B�\)C��)                                    Bx_��\  T          @�G�>��?�Q�@qG�B�p�B���>�׾�\)@|(�B�B�C�Ǯ                                    Bx_�  �          @�  >�?�
=@fffB�u�B�(�>�>u@~{B�\A�\)                                    Bx_��  �          @��=�\)?Ǯ@|(�B��3B��=�\)<#�
@��B�.A�                                    Bx_�)N  �          @��R>W
=?��@n�RB|{B�33>W
=>Ǯ@��B���Bv\)                                    Bx_�7�  �          @��\?�33@*�H@L��B>��B�p�?�33?���@|��B�(�BC�                                    Bx_�F�  "          @��
?p��@:�H@G
=B5p�B�\)?p��?˅@}p�B�\Bm�                                    Bx_�U@  
�          @�p�?O\)@C33@FffB2  B�=q?O\)?��H@\)B�W
B�L�                                    Bx_�c�  �          @��?J=q@L��@Mp�B0��B�B�?J=q?���@��B�\B��                                    Bx_�r�  
�          @�\)?:�H@N�R@W�B6{B�\)?:�H?��
@��B���B�8R                                    Bx_��2  
�          @��\?O\)@`  @N�RB'��B��
?O\)@@���ByQ�B��                                    Bx_���  �          @�p�?��@dz�@Mp�B#(�B�z�?��@
=q@���Br��B��                                    Bx_��~  "          @���?\@|��@9��B	�RB��
?\@(��@�z�BV�\Bo�H                                    Bx_��$  
�          @���?��R@�
=?���A��HB�\)?��R@�=q@P��B�B�R                                    Bx_���  Z          @�p�?�z�@�\)?c�
A(�B�#�?�z�@��\@(Q�A�\B�u�                                    Bx_��p  �          @���?���@���=u?+�B���?���@��R?��
A�(�B��f                                    Bx_��  �          @��?�Q�@�\)��Q�����B��q?�Q�@��?��\AvffB�B�                                    Bx_��  
�          @��
?��
@��\��\����B�.?��
@�\)?�=qAXz�B�B�                                    Bx_��b  
�          @�33?��@��;��R�u�B��?��@��R?��A�{B��{                                    Bx_�  
�          @�=q>�  @�{>\@��HB���>�  @�Q�?�Q�A��B���                                    Bx_��  �          @�G�>��@�p�?�p�A�B���>��@QG�@FffB,��B���                                    Bx_�"T  "          @���?(��@�p�?Q�A'33B�?(��@u�@A�ffB�z�                                    Bx_�0�  �          @�=q?�Q�@�z�=�G�?�33B���?�Q�@��?�z�A��B�L�                                    Bx_�?�  �          @��R?�ff@�Q��G���\)B���?�ff@���?�z�A��
B�\                                    Bx_�NF  
Z          @�=q?.{@��>�?�B�?.{@z�H?�\)A�33B�L�                                    Bx_�\�  
�          @���?&ff@�{>�@ǮB�u�?&ff@p��?�AՅB�u�                                    Bx_�k�  �          @�33?���@{�>\)?��HB���?���@hQ�?\A��RB�W
                                    Bx_�z8  T          @�(�?h��@|�Ϳ.{��
B�
=?h��@{�?B�\A*{B��                                    Bx_���  T          @���>��@�  ����k�B���>��@u�?�A�
=B�Q�                                    Bx_���  �          @�녾�p�@�G�    �#�
B�� ��p�@q�?�A��B�8R                                    Bx_��*  
�          @���?�@����8Q��p�B�(�?�@vff?�G�A�33B�k�                                    Bx_���  "          @�p�?��\@�  ����  B�k�?��\@{�?h��AIB��                                    Bx_��v  �          @�z�?u@~�R�����RB���?u@{�?Y��A>�HB���                                    Bx_��  �          @��
?E�@|�ͿTz��:�\B�\?E�@\)?��A=qB�G�                                    Bx_���  �          @��?��@}p��h���Lz�B���?��@���?�@��
B��                                    Bx_��h  
�          @|��>��@u�5�&=qB�  >��@vff?+�A��B�                                    Bx_��  �          @z=q?�  @hQ�z��	G�B�
=?�  @fff?8Q�A*{B���                                    Bx_��  S          @���?Q�@`�׿ٙ����
B��)?Q�@xQ�\��B�G�                                    Bx_�Z  T          @x��>��R@XQ����RB��>��R@s33����{B���                                    Bx_�*   T          @\(��#�
@P�׿Y���eG�B�{�#�
@Vff>���@�  B�z�                                    Bx_�8�  �          @[�=u@J�H��  ��{B��H=u@Tz�>��@(��B���                                    Bx_�GL  T          @P  ?��@C�
�h�����B���?��@K�>L��@b�\B��                                    Bx_�U�  "          @O\)?   @Dz�h�����B���?   @L(�>B�\@\��B�W
                                    Bx_�d�  �          @;�>W
=@3�
�.{�VffB��{>W
=@,��?O\)A�
=B�8R                                    Bx_�s>  T          @:�H=���@9��=��
?���B�
==���@,��?���A�z�B�                                    Bx_���  �          @.�R�W
=@+��B�\����B���W
=@%�?:�HA|  B�#�                                    Bx_���  T          @=q�\)@  �
=q�P��B���\)@33>��@��B�=q                                    Bx_��0  
�          @Q쾞�R?��xQ��ۅB�G����R@�
��z����B�B�                                    Bx_���  T          @녿333@33?��Aw
=B�녿333?�p�?�G�B(�B�ff                                    Bx_��|  �          @�Ϳ(�?�Q�?�G�Bz�B�k��(�?�z�?�G�BUQ�B�
=                                    Bx_��"  �          @�Ϳ333?�
=?��HBffB�Q�333?�?ٙ�BMG�B�Ǯ                                    Bx_���  �          ?�Q�0��?�G�?}p�A��B��f�0��?��?�Q�B@�B�z�                                    Bx_��n  
�          @G��!G�@ ��?E�A�Q�B�  �!G�?У�?�33B�\B�k�                                    Bx_��  "          @(��>W
=@(�?p��A��B�>W
=?�p�?ٙ�B!��B��q                                    Bx_��  �          @p���@�?��Al(�B�k���?��
?�(�B	G�B��                                     Bx_�`  
�          @�
��=q@�R>�A?�
B��;�=q?�Q�?���A��B���                                    Bx_�#  T          @8Q��(�@4z�>���@���B�Q��(�@#33?�G�A�p�B�#�                                    Bx_�1�            @0  ���H@+�>��@��
B��f���H@(�?��A�{B��f                                    Bx_�@R  
�          @
=q��(�@
=>��@�Q�B��)��(�?�Q�?W
=A�Q�B���                                    Bx_�N�  
�          @0  �(��@(Q�>���@�  B�W
�(��@Q�?�33A�{B�(�                                    Bx_�]�  T          @7
=��R@1�=�G�@33B�W
��R@%�?��
A�G�B�.                                    Bx_�lD  "          @<�Ϳ(�@7
=>\)@0  B���(�@*=q?�=qA���B��                                    Bx_�z�  �          @?\)�^�R@6ff    <#�
B��ÿ^�R@,(�?p��A�\)B��)                                    Bx_���  "          @C33�:�H@<�ͽ�G���Bϣ׿:�H@4z�?^�RA���B��
                                    Bx_��6  
�          @HQ�B�\@AG��.{�C33B���B�\@9��?W
=Ax��B�.                                    Bx_���  �          @A녿O\)@6ff��z����BӔ{�O\)@2�\?(��ALQ�B�=q                                    Bx_���  T          @Dz��(�@AG��B�\�fffB�B���(�@:�H?O\)Au�B���                                    Bx_��(  T          @AG���@<(��\�陚B�W
��@:=q?��A7�BȔ{                                    Bx_���  �          @@�׾�@=p��#�
�HQ�B�k���@6ff?O\)AzffB��                                    Bx_��t  T          @3�
��G�@�&ff�aB�����G�@(�=�\)?���B�(�                                    Bx_��  
�          @<(��#33?�{�s33���Ck��#33?�{�
=q�*{C޸                                    Bx_���  
�          @0  �Q�?�{��G����\Cu��Q�?�\)���/
=Cٚ                                    Bx_�f  �          @1G���?�G����>ffCk���?��ͼ��
�\C�=                                    Bx_�  T          @H�ÿ�@"�\>8Q�@S33B�녿�@ff?}p�A��
B�G�                                    Bx_�*�  �          @B�\�@
�H�#�
�@��C���@ff?��A'�
C�                                     Bx_�9X  �          @C33��
=@  �\)�*�HC���
=@�
>B�\@i��C��                                    Bx_�G�  "          @;����
@ff��G���G�C@ ���
@�
��\)����B��                                    Bx_�V�  T          @E�c�
@7��O\)�rffB�W
�c�
@>�R>��@.{B�#�                                    Bx_�eJ  
�          @I�����@9����\�\)B�W
���@:=q>�G�Ap�B�.                                    Bx_�s�  �          @<(��Ǯ@�����#\)B�\�Ǯ@(�>�\)@�{B�8R                                    Bx_���  T          @+���=q@녿8Q��{�B���=q@	���#�
�c�
B�z�                                    Bx_��<  
�          @)���У�?�p�����QCp��У�@z�=u?��C :�                                    Bx_���  
(          @7���?�{���*�\C
�H��?�Q�=�Q�?��HC	��                                    Bx_���  "          @N�R�z�@�ÿ(��-C�R�z�@p�>8Q�@J�HC(�                                    Bx_��.  
�          @HQ��   @��(��4��C� �   @��>��@/\)Cٚ                                    Bx_���  
_          @G
=��33@ff�333�O�
C �3��33@p�=��
?�z�B��{                                    Bx_��z  T          @AG�� ��@Q�\)�-��CY�� ��@��>\)@%Cs3                                    Bx_��   �          @>�R�޸R@
=��
=�{B���޸R@�>�{@�33B��=                                    Bx_���  "          @>�R�У�@(���(����B�uÿУ�@p�>�Q�@��
B�33                                    Bx_�l  �          @A��   @  ���\)C�=�   @	��?+�AN�HC�H                                    Bx_�  �          @Tz���@#33>�@G�C����@Q�?k�A�=qC�q                                    Bx_�#�  "          @G���@��>���@��C����?�p�?�  A���C	�R                                    Bx_�2^  "          @O\)�333?��H?p��A���C��333?\(�?��\A�\)C"�\                                    Bx_�A  �          @Z�H�7�?��H?��A�ffC0��7�?:�H?�33A��C%�f                                    Bx_�O�  
�          @Z�H�-p�?��?\A�Q�C���-p�?Y��?�\)B�RC"z�                                    Bx_�^P  "          @[��1�?Ǯ?�(�A�{C�3�1�?�{?��A�=qCB�                                    Bx_�l�  T          @\���:�H?�p�?���A���C)�:�H?���?�  A�{C�
                                    Bx_�{�  T          @mp��P��?�=q?��A�33C�R�P��?k�?��RA�{C$.                                    Bx_��B  "          @l���L��?�ff?�  A|(�C��L��?�?�
=A�(�C��                                    Bx_���  "          @tz��]p�?�?�=qA��C!aH�]p�?G�?���A�Q�C'B�                                    Bx_���  �          @j=q�\(�?J=q?n{Am�C'&f�\(�>�?���A�  C,O\                                    Bx_��4  "          @h���Z=q?�ff>Ǯ@���C"�{�Z=q?c�
?0��A2=qC%s3                                    Bx_���  �          @b�\�I��?�논��
�ǮCu��I��?���>��@���Cu�                                    Bx_�Ӏ  �          @e�R�\?�33>�  @~�RC�R�R�\?�G�?(��A)p�C�                                    Bx_��&  T          @k���@z��33��G�C5���@-p��aG��bffC ��                                    Bx_���  �          @mp��G�@(�ÿ����z�B���G�@AG��G��A�B��\                                    Bx_��r  T          @l(��	��@%����
��(�C���	��@:�H�0���,Q�B��R                                    Bx_�  �          @k���@!G������C(���@5��(���
C �H                                    Bx_��  �          @hQ��'�@녿�G���33C
��'�@�R��\)���RC�{                                    Bx_�+d  �          @dz��>{?�\�c�
�g�C33�>{?����������C��                                    Bx_�:
  
�          @g
=�(��@�
�����33C��(��@
=�#�
�$��C
B�                                    Bx_�H�  �          @j=q�{@�׿�p����\C	}q�{@&ff�=p��;�C��                                    Bx_�WV  �          @j�H���@33��ff��Q�C)���@*=q�L���J�RC�                                    Bx_�e�  
�          @e���@
=�����C���@*�H�+��,(�C�                                     Bx_�t�  �          @`����\@p����H��\)C�f��\@#33�@  �Ep�C��                                    Bx_��H  "          @dz��W
==u�n{�v�HC2�f�W
=>��ÿ^�R�f�RC.W
                                    Bx_���  "          @q��k��8Q�L���C
=C6�\�k�=u�O\)�G�C3�                                    Bx_���  
Z          @o\)�`  >�녿�����Q�C-c��`  ?8Q�u�q��C(k�                                    Bx_��:  "          @l(��Z=q>�녿�����ffC-\�Z=q?G���33����C')                                    Bx_���  "          @qG��b�\>����H���
C,���b�\?J=q���\�|Q�C'^�                                    Bx_�̆  T          @o\)�`  >��H��(���
=C,{�`  ?Q녿��
��=qC&�\                                    Bx_��,  
�          @p���b�\>�Q쿚�H��=qC.:��b�\?0�׿�����\C(�3                                    Bx_���  
�          @vff�e�>�z῰����{C/c��e�?+����R��=qC)\)                                    Bx_��x  
�          @s33�_\)=��Ϳ��R����C2\)�_\)?���33���HC+z�                                    Bx_�  T          @hQ��S�
>Ǯ������Q�C-G��S�
?B�\��
=���C'�                                    Bx_��  �          @dz��K�?녿�  ��33C)�H�K�?xQ쿣�
���
C#�                                    Bx_�$j  T          @\���8Q�?��ÿǮ����C���8Q�?�(���Q���33C�3                                    Bx_�3  �          @XQ��%?�
=�˅��\C�%?�=q��\)��{C�=                                    Bx_�A�  �          @aG��!G�?�녿�\��Q�C��!G�@z῞�R��=qC��                                    Bx_�P\  
_          @dz��2�\?E��z��(�C$�f�2�\?��ÿ�\��Q�C�R                                    Bx_�_  �          @_\)�*=q?s33�����C W
�*=q?�p���
=��RC�H                                    Bx_�m�  "          @^{�   ?ٙ���Q��陚C���   @ff��33��z�C�                                    Bx_�|N  "          @Z�H�?��Ϳ�z����
C���@\)������{CE                                    Bx_���  �          @dz����?�
=��p����
C�����@��\)��Q�CE                                    Bx_���  �          @n{�!G�@p��˅����C
���!G�@$z�h���c
=Cs3                                    Bx_��@  T          @q��\)@�ÿ��
��33C(��\)@.{�O\)�EG�Cs3                                    Bx_���  �          @p  ��@��\��\)C����@,�ͿO\)�H��C{                                    Bx_�Ō  
�          @q��   @'
=������Q�CǮ�   @333���R���C�R                                    Bx_��2  �          @r�\��R@"�\�������HCh���R@333���
{C��                                    Bx_���  T          @s33�Q�@�Ϳ���ͅC.�Q�@3�
�h���]��CO\                                    Bx_��~  "          @u��
@����33���C�R��
@4z῕���\CJ=                                    Bx_� $  
�          @s�
�z�@(�����C���z�@+���z���G�C�                                    Bx_��  T          @tz���@%������C ����@>�R�����B��                                    Bx_�p  �          @u����@-p�� ����=qB������@I���������B���                                    Bx_�,  �          @u�
�H@=q��
��C���
�H@8Q쿪=q��z�B��                                    Bx_�:�  �          @w���@ff��
�z�C�q��@8�ÿ˅��B��                                    Bx_�Ib  "          @vff��G�@���
=�
=B�z��G�@@  ��{��33B���                                    Bx_�X  �          @tz��  @$z��{�33B��{��  @DzΌ����z�B�ff                                    Bx_�f�  �          @u����H@*�H�Q��=qB�W
���H@H�ÿ�=q��\)B�.                                    Bx_�uT  �          @s33��G�@333��
�B���G�@P  ���R����B��                                    Bx_���  "          @u���@*�H���H��
=B�\)���@Fff�����B��                                    Bx_���  T          @u��@��ff�Q�C	{�@*=q������{Cc�                                    Bx_��F  �          @u�\)@�R�
=q�	  C#��\)@.{���R���HCz�                                    Bx_���  �          @o\)��
?��
�ff�z�Ck���
@���ff��p�C�3                                    Bx_���  �          @j=q�	��?�����
���CE�	��@�ÿ޸R��C�                                    Bx_��8  �          @U����?��������C����?˅�����HC�{                                    Bx_���  �          @hQ��
�H?�=q���  C�)�
�H@���(��ᙚC��                                    Bx_��  �          ?�녿Y��>8Q�&ff��
C'ٚ�Y��>�33�z�� �
C�)                                    Bx_��*  
�          ?��;8Q�5?Q�B@z�C��8Q�fff?(�B�\C�w
                                    Bx_��  �          ?@  >k�>�>���B`��A�
=>k�<��
>��Brz�@�(�                                    Bx_�v  �          ?p��?�\>��?=p�BXp�A}�?�\�L��?@  B]�C�4{                                    Bx_�%  �          ?k�>�z�>.{?O\)B�G�A�33>�z�#�
?Q�B�L�C�ٚ                                    Bx_�3�  �          ?���>8Q�?
=?O\)BQ��B��q>8Q�>�p�?p��B�B}�                                    Bx_�Bh  �          ?�z�k�?�\)?�B6G�B��)�k�?G�?�Bp
=B�8R                                    Bx_�Q  �          @p��
=?���?L��A���B��
�
=?�
=?��\B�B��H                                    Bx_�_�  �          @(��
=q@�>���A'\)B���
=q?�z�?h��AŮBӏ\                                    Bx_�nZ  �          @\)�333@�>�p�A
�RBՊ=�333@
=q?s33A�B�
=                                    Bx_�}   �          @7
=�
=q@.{?   A#\)Bʞ��
=q@   ?�Aď\B̅                                    Bx_���  �          @@�׿Q�@8��>B�\@g
=Bӊ=�Q�@0  ?h��A�G�B�{                                    Bx_��L  �          @C33�z�H@6ff>�z�@�  B�녿z�H@+�?�  A�  B�\                                    Bx_���  
�          @G
=��Q�@7�>\)@$z�B�.��Q�@/\)?Y��A\)B�
=                                    Bx_���  �          @7��k�@-p�<#�
>\)Bـ �k�@'�?0��A]p�Bڮ                                    Bx_��>  T          @8Q�W
=@/\)�L������B���W
=@-p�>��HAz�B֏\                                    Bx_���  �          @8�ÿn{@,�;�p���33B���n{@-p�>��
@ȣ�B��H                                    Bx_��  �          @@  �^�R@2�\�
=�6�RB֔{�^�R@6ff=�@�
B��f                                    Bx_��0  �          @E����@2�\�(��7\)B�ff���@7
==���?���B�z�                                    Bx_� �  �          @C33��G�@4z���z�B�z῁G�@7
=>B�\@j=qB���                                    Bx_�|  �          @H�ÿ�p�@.{�h����{B�uÿ�p�@7��aG��~{B�W
                                    Bx_�"  �          @Fff���R@������p�B�.���R@&ff�G��h��B                                    Bx_�,�  �          @G
=���R@*�H�Tz��|��B��f���R@2�\�#�
�<(�B���                                    Bx_�;n  �          @G����
@6ff�+��G�
B۸R���
@;�=#�
?G�B�                                    Bx_�J  �          @HQ쿌��@;��
=q��B������@>{>B�\@\(�B܏\                                    Bx_�X�  T          @AG���  @�\����ϙ�C�׿�  @�\�E��p��B���                                    Bx_�g`  T          @>�R��(�?�p�����\CW
��(�?����\�ՙ�C�                                    Bx_�v  T          @L�Ϳ�Q�@
=��G����B�(���Q�@!녾����\)B�W
                                    Bx_���  "          @Mp��\@2�\������B���\@3�
>�\)@�{B�3                                    Bx_��R  �          @HQ쿦ff@5�B�\�aG�B�aH��ff@333>��HAz�B��                                    Bx_���  �          @E�����@2�\>�Q�@�p�B➸����@(Q�?�G�A�(�B�8R                                    Bx_���  �          @?\)��  ?У׿h������C���  ?�ff���H�=C�f                                    Bx_��D  �          @B�\��G�?��������C .��G�@�\������HB���                                    Bx_���  �          @G����H@녿�����B�#׿��H@%��u��  B��f                                    Bx_�ܐ  T          @E��p�@���{��\)B�33��p�@   ��ff��G�B�.                                    Bx_��6  �          @Dzῼ(�?�33��z��{B�����(�@�\��z���p�B�k�                                    Bx_���  �          @HQ쿼(�@z῾�R��p�B��3��(�@&ff�fff���\B��H                                    Bx_��  �          @HQ쿴z�@{�k���
=B�k���z�@'
=���
��33B잸                                    Bx_�(  �          @N�R���@@��>aG�@~�RBݣ׿��@8Q�?k�A�  B�ff                                    Bx_�%�  �          @I���333@C�
��Q��p�B���333@@  ?(�A4��B�Q�                                    Bx_�4t  T          @J=q�aG�@C�
=L��?^�RB�ff�aG�@=p�?@  Ab�\B���                                    Bx_�C  �          @AG�>aG�@,��?��AΏ\B��)>aG�@�?�33B=qB�ff                                    Bx_�Q�  �          @5�=�G�@
=?��B�B���=�G�?���@�\B9
=B��R                                    Bx_�`f  �          @333>�{@��?˅B�B�z�>�{?��
@33BB(�B�k�                                    Bx_�o  �          @
=?J=q?B�\?�p�BU�HB/�?J=q>��?�\)Bt�A�33                                    Bx_�}�  �          ?��?��ý���?\B[C�Y�?��þ�G�?��HBN��C���                                    Bx_��X  �          @(�?�  �8Q�?�=qB8C�}q?�  �
=q?�  B,��C�+�                                    Bx_���  �          @<(�?��!G�@
=qB;��C��?�����?��HB&33C�\                                    Bx_���  �          @:�H?�33��@
=qB?�RC��{?�33�s33@   B-z�C���                                    Bx_��J  �          @<��?�\)��Q�@��BI��C���?�\)���@(�B@G�C�*=                                    Bx_���  T          @1G�?�ff>8Q�@BD�
@��
?�ff��z�@�BCffC�u�                                    Bx_�Ֆ  �          ?�33����?�(��u��B�#׾���?���>�=qA"�RBͅ                                    Bx_��<  �          @Q쿼(�?����Q����C
��(�?��ÿ�\)�ۙ�C�                                    Bx_���  �          @5���
=?޸R��
=��p�C
=��
=@ �׿�G���G�C�H                                    Bx_��  
�          @0  ��G�?�׿�����\C �3��G�@��W
=��33B��)                                    Bx_�.  �          @1녿�33@ff��33���B�W
��33@33�(���\z�B�                                    Bx_��  T          @8Q쿴z�@�ÿ����G�B���z�@��Q�����B�                                    Bx_�-z  �          @3�
��ff@�
������z�B�B���ff@z�fff��B�z�                                    Bx_�<   �          @1녿�{?�p���33�Q�C	Y���{?�������HC�3                                    Bx_�J�  �          @;���?��������C�=��@�u���C��                                    Bx_�Yl  �          @2�\�33?�{�\�
=C}q�33?����R�ӮCff                                    Bx_�h  �          @3�
����?��ÿ�33��HC�����?�녿�=q��C
s3                                    Bx_�v�  �          @-p���p�?����\)��
CJ=��p�?�{�����{C	
                                    Bx_^  �          @*=q��Q�?c�
��=q���Cs3��Q�?����������HC8R                                    Bx_  �          @!녿�=q?�녿�����C
=��=q?�33�������C��                                    Bx_¢�  �          @+���p�?�\)��z���\)Cp���p�?�33����ȸRC�{                                    Bx_±P  T          @Dz����?B�\�޸R�  C"W
���?�{�����Q�C\                                    Bx_¿�  �          @Dz����?�G��������C�����?�������33C��                                    Bx_�Μ  �          @B�\�	��?�p��\���C��	��?�G������RC��                                    Bx_��B  �          @E�"�\?녿���  C'c��"�\?fff���R���C ��                                    Bx_���  �          @Fff�'�?(��\���
C&��'�?h�ÿ�{����C ��                                    Bx_���  �          @8���!G�?&ff��(����HC%�)�!G�?c�
������HC ��                                    Bx_�	4  �          @A��.{>���33��G�C*T{�.{?0�׿��
��\)C%��                                    Bx_��  �          @I���7�=u��G����C2޸�7�>��
��(���G�C-��                                    Bx_�&�  
�          @5��%��B�\�Tz�����CD� �%��z�xQ����RC@��                                    Bx_�5&  �          @/\)�{�xQ��ff��CI���{�\(��#�
�]p�CG0�                                    Bx_�C�  �          @0  �!녿Y�����3�CF�H�!녿:�H�0���k�
CD�                                    Bx_�Rr  T          @/\)�!G�?\)�L����  C'���!G�?5�+��f=qC$W
                                    Bx_�a  �          @.{� ��?��
�Ǯ��C
W
� ��?�=q���
�ǮC	�                                    Bx_�o�  "          @!G���{?�(����R�\B�3��{?���>�=q@��
B�\                                    Bx_�~d  T          @=q��=q?�G��@  ��{CxR��=q?��׾��L��C
��                                    Bx_Í
  "          @%���{?����=q�=qCJ=��{?�=q�����Q�C�{                                    Bx_Û�  �          @ff�(��@��.{����B�#׿(��@
�H��  �ʏ\Bճ3                                    Bx_êV  �          @���#�
@p���G��+�
B�p��#�
@  <#�
>���B��)                                    Bx_ø�  �          @*=q��@'
=����P��B�W
��@%>�p�A�RB�ff                                    Bx_�Ǣ  �          @#�
�333@�
��=q���B�Ǯ�333@z�>B�\@�Q�Bը�                                    Bx_��H  �          @33��  ?�z�#�
���
C8R��  ?�z�>.{@��C:�                                    Bx_���  �          @�H?Q�?\(��W
=�\)B8��?Q�?�G��(������BJ��                                    Bx_��  T          @p�?5?�p���Q��뙚B�� ?5@��G���=qB�Ǯ                                    Bx_�:  
�          @1G�=�\)@$z�h�����B��=�\)@,�;Ǯ�ffB�{                                    Bx_��  �          @333�u@(�������B�LͿu@��u��=qB�33                                    Bx_��  
�          @*=q�Tz�?�\���
�)(�B�ff�Tz�@���33� 33Bߞ�                                    Bx_�.,  "          ?�
=�h��?�������B���h��?�(��E���p�B�\)                                    Bx_�<�  T          @�׾�\)?��=��
@�BŔ{��\)?޸R>�
=AU�B�\                                    Bx_�Kx  T          @!G�?��H?:�H?�(�BG�A�ff?��H>�(�?˅B'z�Ac\)                                    Bx_�Z  �          @
=?�{?s33?��RB'z�B�\?�{?&ff?��B=�RAˮ                                    Bx_�h�  �          @?�(�?�  ?�ffB=qB!�?�(�?z�H?\B"�BG�                                    Bx_�wj  �          @�?��R?aG�?�(�B!(�A�ff?��R?
=?�{B4ffA��                                    Bx_Ć  T          @33?�
=?W
=?�
=BQ�A�G�?�
=?��?��B �HA��R                                    Bx_Ĕ�  T          ?���=�\)?�(�?Q�BB���=�\)?�ff?��B3
=B���                                    Bx_ģ\  �          ?��H?.{?�ff?:�HA�33Bd=q?.{?c�
?h��B  BS{                                    Bx_Ĳ  �          @%�?���?��
?��
A��BQ�?���?��\?��RBG�A�                                    Bx_���  "          @#33�0��@����
��  B��0��@\)�!G��p��Bօ                                    Bx_��N  �          @(Q�W
=@33������B��
�W
=@�׿k���  B�                                    Bx_���  �          @@�׿�{?��
���H�%ffCaH��{?�\)����z�C�q                                    Bx_��  T          @P  �?�녿������Cٚ�?�zῧ����C	��                                    Bx_��@  
�          @N�R�{?��R����\)C=q�{?��
��z����C5�                                    Bx_�	�  
�          @S33�Q�?\(��   �  C ��Q�?��H���(�C                                    Bx_��  �          @[����=#�
����7  C3\���>��H�=q�2p�C(��                                    Bx_�'2  
�          @X�����>W
=��
�,�HC/\���?!G���R�%��C%��                                    Bx_�5�  �          @P���G�?�(���������Cٚ�G�?�Q쿂�\���RC�=                                    Bx_�D~  T          @@�׿���@\)>L��@}p�C�
����@
�H?��A>ffC�                                    Bx_�S$  �          @8Q��p�?��>��
@�
=C���p�?��?�A733C��                                    Bx_�a�  "          @@�׿���@!�>k�@�
=B�33����@(�?.{Aa��B垸                                    Bx_�pp  �          @>{��p�@
=q�L���z=qCk���p�@
�H>B�\@j�HCff                                    Bx_�  T          @@  ��G�@\)�\)�/33B���G�@#33��G���B�L�                                    Bx_ō�  �          @:=q���R@����ffB�
=���R@=q>��R@ʏ\B�ff                                    Bx_Ŝb  �          @7
=���?��H��\)���C�{���?�Q�>u@���C{                                    Bx_ū  "          @7��(��?p�׾�\)��ffC \)�(��?xQ��G��\)C�3                                    Bx_Ź�  
�          @<(��-p���p����\��33C;��-p��B�\�����p�C7�                                    Bx_��T  T          @?\)�<��<#�
����
C3�q�<��=��;�����C2�                                    Bx_���  T          @<���;������
=C6���;���G��\)�1�C6&f                                    Bx_��  
(          @>�R�;�>�p��u���C,�{�;�>��;.{�S�
C,&f                                    Bx_��F  T          @A��0  ?Tz�\)�/�C#33�0  ?h�þ����
=C!��                                    Bx_��  
�          @AG��{?�Q�?
=A3�
C
��{?�?h��A��HC�H                                    Bx_��  
�          @AG��/\)?z�?��AB�HC(��/\)>��?0��A`z�C*T{                                    Bx_� 8  T          @A��>�R=#�
>��@��
C35��>�R��>��@�z�C4��                                    Bx_�.�  T          @A��/\)?L�Ϳ
=q�+�
C#�f�/\)?aG��������C")                                    Bx_�=�  T          @?\)�*�H?�p�>�A{CQ��*�H?���?(��AM�C�                                    Bx_�L*  T          @<(��#�
?�\)>�AC�H�#�
?��\?0��AX��C�)                                    Bx_�Z�  
�          @@  �.{?�{>��
@�  C��.{?��?   A��C
=                                    Bx_�iv  �          @@��� ��?�p�?5A\Q�Cff� ��?��?s33A�  C�{                                    Bx_�x  T          @G��'�?�(�?O\)Ar�\C���'�?���?�ffA�\)CL�                                    Bx_Ɔ�  �          @=p��?�{?�
=A�{C���?�33?��A�(�C޸                                    Bx_ƕh  T          @8�ÿ�  ?�
=?�  A��C+���  ?�z�?�  B��C(�                                    Bx_Ƥ  T          @1G���
=?�G�?��RB��C��
=?J=q?��B!�
C
=                                    Bx_Ʋ�  
�          @.�R?�(�?
=q?��B=qAs�?�(�>��?��HB"�R@�Q�                                    Bx_��Z  "          @;�?�  ?+�@�\BY{A�\)?�  >�z�@�Bd�\A/\)                                    Bx_��   �          @)��?�G��ٙ�?z�HA���C�޸?�G���=q?0��A��C��                                    Bx_�ަ  "          @J�H@.�R>B�\?���A�z�@~{@.�R��?�=qAиRC��{                                    Bx_��L  T          @a�@AG�>.{?��A���@H��@AG���?��A��C�Ф                                    Bx_���  
�          @j�H@I������?��A�ffC��@I�����?�A�ffC�T{                                    Bx_�
�  
�          @u@Q녾u?�Q�A��HC��@Q녿��?��A�  C�>�                                    Bx_�>  �          @U�@.�R����?�  A�ffC�XR@.�R���?��A�33C�Y�                                    Bx_�'�  
�          @s33@N{<�@   A�G�?�@N{��z�?�p�A�Q�C�l�                                    Bx_�6�  �          @���@X��>�z�@	��Bp�@�33@X�ýu@
�HB�C��H                                    Bx_�E0  
�          @aG�@L(���=�Q�?\C��f@L(������Ϳ�\)C��                                    Bx_�S�  T          @s33@hQ�>#�
?��
A|z�@"�\@hQ�    ?��A�C���                                    Bx_�b|  �          @q�@e�>��
?�\)A�ff@�G�@e�>\)?�z�A���@�                                    Bx_�q"  "          @L(�@E�?�?
=A)Ap�@E�>�G�?(��A@(�A ��                                    Bx_��  T          @e�@O\)?n{?�ffA���A�ff@O\)?B�\?�
=A���ARff                                    Bx_ǎn            @�(�@\(�?�=q?�p�A�  A��@\(�?B�\@�A�z�AG�                                    Bx_ǝ  
�          @��\@Tz�?��H@ ��A��A��@Tz�?aG�@
�HB  Am�                                    Bx_ǫ�  �          @P��?��R@�?�ffA��B6�?��R?�G�?�B
=B&33                                    Bx_Ǻ`  �          @G
=?�Q�?��?�  A���B0p�?�Q�?У�?�\B(�B                                       Bx_��  "          @b�\@#33?��R?�Q�BG�A��@#33?�
=@	��B��A�=q                                    Bx_�׬  �          @g�@/\)?Ǯ?�  A�z�A��@/\)?��
?�(�BQ�Aǅ                                    Bx_��R  �          @o\)@*�H?�  @�B(�A���@*�H?fff@ffB=qA�                                    Bx_���  �          @z=q@/\)?��H@�RB�
A�z�@/\)?Q�@(��B*�A�33                                    Bx_��  �          @aG�@(Q�?�p�@ ��B
��A�  @(Q�?h��@
�HB  A��\                                    Bx_�D  �          @]p�@/\)?�
=?޸RA�A�z�@/\)?h��?�33B�A�z�                                    Bx_� �  
�          @��R@E�?��@!G�B�\A�{@E�?s33@,(�B�\A��H                                    Bx_�/�  �          @���@XQ�?�=q@#�
B
  A��@XQ�?�
=@1G�B�A��                                    Bx_�>6  �          @�G�@\��?�Q�@�HB �
A�Q�@\��?��@)��Bp�A�(�                                    Bx_�L�  �          @�G�@J=q?�
=@�B  A�Q�@J=q?���@%B��A���                                    Bx_�[�  �          @���@L(�?�=q@(�A�{A�\@L(�?�p�@�B
��A�                                    Bx_�j(  �          @�{@HQ�@�
?��A�  B�\@HQ�?�G�@
�HA�33A��                                    Bx_�x�  �          @�=q@c33?��H?�(�A�A��\@c33?�33@
=qA�
=A��                                    Bx_ȇt  �          @�{@c33?��?�G�AǮA�p�@c33?���?�Q�A�33A�                                      Bx_Ȗ  �          @��\@`��?�33?�(�A�G�A��H@`��?c�
?�{A��
Ab�H                                    Bx_Ȥ�  T          @�33@c�
?��?�Q�A�=qA�p�@c�
?aG�?�=qA�ffA]�                                    Bx_ȳf  �          @tz�@\(�?�G�?���A�{A�33@\(�?��?��RA���A�G�                                    Bx_��  �          @s33@e�?���?z�A�A�@e�?��
?:�HA1��A���                                    Bx_�в  "          @c�
@U�?�=q?!G�A"�\A���@U�?z�H?E�AIp�A��H                                    Bx_��X  "          @4z�?�{?��H?��A�=qB�?�{?�G�?�p�B�B��                                    Bx_���  �          @333@   ?h��?�Q�B�
A�G�@   ?(��?�B"A���                                    Bx_���  T          @Z�H@C�
?aG�?�Q�A��RA�  @C�
?333?��A���AN{                                    Bx_�J  T          @`  @P  ?Q�?uA~ffAa�@P  ?+�?��A��A:�H                                    Bx_��  T          @Mp�@@  ?G�?O\)Ai��Ai�@@  ?(��?h��A��AF{                                    Bx_�(�  �          @Q�@B�\?O\)?k�A��HAn=q@B�\?+�?��
A�(�AG33                                    Bx_�7<  
�          @U@G�?&ff?xQ�A�33A;�@G�?�\?�ffA��HA�                                    Bx_�E�  
�          @U@H��?(�?uA�{A/33@H��>��?��A��HA�                                    Bx_�T�  �          @W�@L(�>�ff?xQ�A�{AG�@L(�>��R?��\A��@�(�                                    Bx_�c.  �          @\(�@N{>��?���A�\)A  @N{>��R?�
=A��H@�                                    Bx_�q�  "          @Z=q@J=q>�ff?���A�33A ��@J=q>�\)?�  A�z�@���                                    Bx_ɀz  
�          @J�H@>�R>L��?}p�A���@x��@>�R=u?�G�A���?�                                     Bx_ɏ               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_ɝ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_ɬl              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_ɻ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_�ɸ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_��^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_�!�  �          @8Q�@�R?��?p��A�ffA�p�@�R?fff?���A�z�A�{                                    Bx_�0B  	�          @8��@$z�?��?
=A?�A�{@$z�?s33?8Q�Ak33A��R                                    Bx_�>�  T          @C33@*�H?�\)>��@��AظR@*�H?�ff?z�A1��A�                                    Bx_�M�  
�          @N{@.�R?�(�?Tz�An�\A�@.�R?���?�G�A��RA�                                    Bx_�\4  T          @W
=@<(�?�=q?B�\AS
=A�G�@<(�?�(�?k�A�\)A��H                                    Bx_�j�  �          @S33@7�?�=q>#�
@/\)A��@7�?�ff>�Q�@��A���                                    Bx_�y�  T          @N{@,(�?��?�A��A�ff@,(�?Ǯ?8Q�AQ��A�Q�                                    Bx_ʈ&  "          @N{@0��?���>�\)@��\A�@0��?\>�A33A�G�                                    Bx_ʖ�  
�          @N�R@(Q�?�  ?�RA1p�B��@(Q�?�z�?W
=Aq��B �H                                    Bx_ʥr  �          @b�\@6ff?�(�?z�A��B
�\@6ff?��?Tz�AZ�\B�\                                    Bx_ʴ  
Z          @U�@��@��?Y��Alz�B'
=@��@ ��?�{A���B 33                                    Bx_�¾  
�          @dz�@/\)@�?J=qAL��B��@/\)@   ?�ffA�\)B\)                                    Bx_��d  "          @Z�H@�\@�\?���A��
B&@�\?�?�Q�A�p�B(�                                    Bx_��
  
�          @aG�@Q�?�33?�p�A���B33@Q�?�z�?��HB��B                                    Bx_��  
�          @b�\@333?޸R?n{A{
=A��\@333?�{?��A��A��H                                    Bx_��V  
�          @g
=@G�?���=L��?J=qA�@G�?�ff>�\)@���A�p�                                    Bx_��  �          @a�@C�
?�z�\��p�A�{@C�
?ٙ��.{�-p�A�                                      Bx_��  �          @\(�@;�?�p���  ��A��@;�?޸R���
=qA��                                    Bx_�)H  �          @Z=q@0  ?���(���Q�B�@0  ?��H�B�\�I��B
=                                    Bx_�7�  
�          @I��?���@5�?0��AM�B��?���@.�R?��A��
B�.                                    Bx_�F�  T          @X��@{@�?
=A%�B;�@{@�?aG�Av�HB6��                                    Bx_�U:  
�          @c�
@=p�?�>�=q@��RA��@=p�?�\>��HA��A�Q�                                    Bx_�c�  
�          @\(�@=p�?˅�\)��\A�=q@=p�?�33��Q�����A��H                                    Bx_�r�  T          @\(�@:�H?�G����
���
A�ff@:�H?�33��\)���\Aͅ                                    Bx_ˁ,  
�          @W�@p�?�=q@ ��B33A�33@p�?Q�@�BG�A�Q�                                    Bx_ˏ�  �          @Tz�@33?s33@
�HB$ffA��@33?.{@G�B-�RA�(�                                    Bx_˞x  
�          @Y��@�
?}p�@�B)  A���@�
?5@��B2��A�33                                    Bx_˭  "          @\��@
=?���@   B:ffA��@
=?B�\@'
=BEffA��R                                    Bx_˻�  "          @`  ?�ff>8Q�@FffB}(�@�z�?�ff�B�\@FffB}
=C�s3                                    Bx_��j  
�          @W�?\(��
=@L(�B�ǮC��)?\(��xQ�@FffB���C��=                                    Bx_��  �          @G
=?xQ�p��@1G�B�  C��?xQ쿡G�@(Q�Bkp�C��3                                    Bx_��  T          @4z�?�=q��\)@�B���C�Ф?�=q���@�Bw�HC�y�                                    Bx_��\  �          @@��?�G�?c�
@G�BDQ�A���?�G�?(�@
=BO
=A�(�                                    Bx_�  "          @A�?��R?@  @	��B5(�A��H?��R>��H@{B==qA]�                                    Bx_��  "          @C33?���?�  @
=B0  Aٮ?���?=p�@{B;{A�p�                                    Bx_�"N  
�          @^{@
=q?��@B+ffA�z�@
=q?��
@�RB8=qA�                                    Bx_�0�  
�          @dz�@&ff?�p�?�33BA�ff@&ff?��R@�
B33A�(�                                    Bx_�?�  �          @l(�@)��?�G�?�\)A�B�R@)��?��
@�
B��A�R                                    Bx_�N@  "          @l(�@   ?��?���A�
=B��@   ?��@	��B�BG�                                    Bx_�\�            @l(�@�
@'�?��A�Q�BOff@�
@�?�=qABF��                                    Bx_�k�  �          @j�H@0��?�{?\A�Q�B
=@0��?�?�(�A�
=A��
                                    Bx_�z2  
�          @n{@3�
@��?�\)A��RB{@3�
@   ?�{A�(�Bz�                                    Bx_̈�  
�          @k�@�R@(��?�AQ�B:�@�R@#�
?Tz�AP��B7z�                                    Bx_̗~  	�          @a�?��@7
=?W
=A\Q�BbG�?��@/\)?�z�A��
B]�                                    Bx_̦$  
�          @aG�?�z�@8Q�?�A�Ba�R?�z�@333?c�
AlQ�B^��                                    Bx_̴�  "          @g�?@  @`��>W
=@X��B��?@  @]p�?(�A��B���                                    Bx_��p  T          @aG�?=p�@Z=q���
���B��=?=p�@Z�H=�\)?��B���                                    Bx_��  
�          ?�(��=p���\)����=qClJ=�=p�����0�����Cj��                                    Bx_��  
(          @,(�?�33?&ff?�G�B �A��?�33?   ?�=qBp�AmG�                                    Bx_��b  "          @p  @0��?�  @z�B=qA�
=@0��?:�H@�HB ��Al��                                    Bx_��  �          @�p�@U�?��@]p�B2�A���@U�?&ff@c33B9G�A/\)                                    Bx_��  
�          @���@N�R?�Q�@]p�B4�
A��@N�R?J=q@dz�B<ffA\��                                    Bx_�T  
�          @�ff@X��?�{@5�B
=A˅@X��?��
@?\)B(�A�(�                                    Bx_�)�  T          @�  @u?�
=@J=qB
=A�\)@u?���@Tz�B\)A��
                                    Bx_�8�  "          @��\@l��?�@>�RB=qA�ff@l��?�=q@I��B�
A�                                      Bx_�GF  �          @��
@^{>�@UB.�H@��
@^{=�Q�@W�B0�?��R                                    Bx_�U�  
�          @��@k�?z�@_\)B,��Aff@k�>B�\@a�B/33@9��                                    Bx_�d�  
�          @�ff@^{?Tz�@[�B/��AW�@^{>�ff@`  B433@�                                    Bx_�s8  
�          @��@g�?u@Z=qB)G�AmG�@g�?z�@_\)B.�A��                                    Bx_́�  
�          @��
@mp�?h��@Z�HB'=qA]p�@mp�?�@_\)B,{A�H                                    Bx_͐�  �          @�{@w
=?k�@UB G�AV�\@w
=?��@Z�HB$��A��                                    Bx_͟*  �          @�@w
=?�  @S�
B�Ah��@w
=?!G�@Y��B#�HAG�                                    Bx_ͭ�  T          @�ff@s�
?���@U�B(�A��@s�
?Tz�@\(�B%�\ADQ�                                    Bx_ͼv  �          @���@\)?���@S�
B\)Aw\)@\)?=p�@Z=qB   A'�                                    Bx_��  T          @�  @z�H?���@L(�B�A���@z�H?��
@Tz�B=qAk�                                    Bx_���  �          @�p�@p��?�{@A�B�A�z�@p��?\@Mp�BQ�A�(�                                    Bx_��h  "          @���@}p�?�  @H��BQ�A�{@}p�?�33@R�\B(�A��
                                    Bx_��  �          @�ff@~�R?�(�@>�RB33A�Q�@~�R?��@G�B�A�(�                                    Bx_��  
�          @�@tz�@
=q@/\)A��A�(�@tz�?�{@=p�Bp�A�G�                                    Bx_�Z  �          @���@N{@�@�A�p�B33@N{@�@=qBffB
=                                    Bx_�#   T          @|��@7
=@{?��A��Bp�@7
=@�?�\)A�ffB                                    Bx_�1�  
�          @�@fff@{?��RA�{A�33@fff?��R@{A���A�                                    Bx_�@L  
�          @�p�@[�@Q�@33A�33B
�R@[�@��@�\A�{A��
                                    Bx_�N�  
(          @�  @e@.{@�A�33B�@e@{@p�A�B
(�                                    Bx_�]�  	�          @�Q�@j=q@1�?��HA���B
=@j=q@#�
@  AծB�H                                    Bx_�l>  
�          @�G�@a�@<(�@z�A£�B{@a�@,��@�A��HB�R                                    Bx_�z�  �          @��H@U@%�@
=A陚B��@U@z�@'�B�B
�                                    Bx_Ή�  �          @�G�@W
=@��@�A�Bff@W
=@Q�@*�HBp�B��                                    Bx_Θ0  
�          @��
@b�\@��@p�A���B{@b�\?��R@+�B��A�ff                                    Bx_Φ�  T          @�p�@g�@z�@��A��HB�@g�@�
@'�B �HA���                                    Bx_ε|  
�          @�\)@e@
=@ ��A�Q�B�@e@@/\)B��A�                                    Bx_��"  �          @��@\��@ ��@�A��
B{@\��@\)@+�B�B                                      Bx_���  �          @�
=@X��@%@"�\A��\Bz�@X��@�
@333B	G�B	{                                    Bx_��n  T          @���@XQ�@#33@{A��B{@XQ�@�@.{B�
B�                                    Bx_��  
�          @�33@R�\@'
=@��A�{B��@R�\@ff@*=qBffB�                                    Bx_���  "          @�z�@U@333@(�A���B��@U@#�
@p�A�RB�\                                    Bx_�`  T          @�
=@XQ�@;�@
=qA��B#�@XQ�@,(�@��A�=qB�                                    Bx_�  �          @�33@Z�H@0  @ ��A�z�Bz�@Z�H@!�@�A�B(�                                    Bx_�*�  �          @��R@E@5�?��RA��\B)��@E@*=q?�\A�  B"��                                    Bx_�9R  �          @��R@7
=@"�\?��\A��\B&p�@7
=@��?\A�p�B�
                                    Bx_�G�  �          @��@Mp�?�z�@��A�=qA�(�@Mp�?�
=@z�B��A�33                                    Bx_�V�  �          @|��?�p�@c33?^�RAP��B��q?�p�@\(�?�p�A���B���                                    Bx_�eD  �          @}p�?�{@I��?�G�A�
=Bm�?�{@>�R?���A�33BhG�                                    Bx_�s�  �          @w�?��@P  ?���A��
B�W
?��@E?ٙ�A�  B~33                                    Bx_ς�  �          @w�@�@'�?�(�A��
BP@�@=q@�RB{BG33                                    Bx_ϑ6  �          @y��@)��@�R?�z�A�B,z�@)��@ff?�z�A�33B&Q�                                    Bx_ϟ�  �          @w�@@�@G�B�\B@�\@@@\)B �B4
=                                    Bx_Ϯ�  	�          @u�@�?�\)@ ��B$��B"�
@�?�{@,(�B3(�B�H                                    Bx_Ͻ(  �          @~�R@!�?�\)@(�BffB=q@!�?�\)@'�B$��Bz�                                    Bx_���  T          @~�R@'
=?�G�@#33B!  A�
=@'
=?�  @,(�B+�HȀ\                                    Bx_��t  
�          @���@,(�?У�@-p�B#�A��@,(�?���@7
=B.Q�A���                                    Bx_��  
�          @��@*=q?˅@*=qB"�RA��@*=q?��@3�
B-��A�ff                                    Bx_���  T          @��@Dz�?k�@(��B�RA�  @Dz�?&ff@.{B#p�A@z�                                    Bx_�f  �          @���@X��@	��?�{A�{B�R@X��@   ?���A�=qA��H                                    Bx_�  �          @��\@U@%?k�AEG�B
=@U@�R?�
=A|��Bp�                                    Bx_�#�  �          @��@l��@�R?���Ayp�A���@l��@ff?�A��A�R                                    Bx_�2X  �          @�G�@{�@   ?���Af{A�@{�?��?�=qA�\)A�=q                                    Bx_�@�  	�          @�Q�@W
=@ ��?�33A���B�@W
=@�?��A�ffB
��                                    Bx_�O�  
�          @��@Tz�@#�
?���A���Bff@Tz�@��?�A�(�B                                    Bx_�^J  
�          @�  @\(�@Q�?�A��B
z�@\(�@��?�33A�33B=q                                    Bx_�l�  �          @���@[�@ ��?�G�A���B  @[�@ff?�G�A�
=B	�R                                    Bx_�{�  
�          @�=q@^�R@��?��A��
B�@^�R@�
@
=A�  A�\)                                    Bx_Њ<  "          @��@S33@{@z�A��
B�R@S33@ ��@�A�z�A�(�                                    Bx_И�  �          @�
=@J�H@��@G�AׅBz�@J�H@(�@  A�=qB
ff                                    Bx_Ч�  
�          @��?�z�@n{?�33A�  B��H?�z�@c�
?�G�A�\)B�                                      Bx_ж.  �          @�=q@33@n�R?У�A�Bt�@33@c33?��RA�\)Bo�
                                    Bx_���  "          @�{@ ��@Y��?�(�A��
Bm��@ ��@L(�@33A��\Bg�                                    Bx_��z  
�          @�z�@,��@�@�HB�HB((�@,��@�@)��B��B��                                    Bx_��   T          @��H@Q�@�\@;�B,{B"�\@Q�?�  @G�B:33B�                                    Bx_���  �          @�Q�?��@:�H@7
=B)z�B�z�?��@(Q�@H��B>ffB�.                                    Bx_��l  "          @�  ?p��@Fff@0  B!G�B�W
?p��@4z�@B�\B6�
B�#�                                    Bx_�  �          @�Q�?�ff@?\)@/\)B =qB�?�ff@,��@AG�B4B��                                    Bx_��  T          @���?��
@7�@1G�B!\)Bw��?��
@%@A�B5
=BmQ�                                    Bx_�+^  "          @�Q�@!�@	��@!G�B�B!G�@!�?��@.{B#  B��                                    Bx_�:  T          @�G�@C�
@(�@�\A�  B�@C�
?�p�@  A�\)B�H                                    Bx_�H�  T          @�Q�@e?�p�?�Q�A�{A���@e?�ff?���A�ffA��
                                    Bx_�WP  T          @��@mp�?��?���A���A��H@mp�?��R?�=qA�A��\                                    Bx_�e�  �          @��@=p�@z�@	��A��
B��@=p�?���@BG�A��                                    Bx_�t�  T          @�{@@��@��?�\A�p�B��@@��@�?�p�A�B
p�                                    Bx_уB  
�          @���@Mp�@
=q?�G�A�\)B��@Mp�?�p�?��HAۮA��H                                    Bx_ё�  "          @���@
=@P��?�G�A�{BXQ�@
=@Fff?���A�{BR�                                    Bx_Ѡ�  
�          @���@   @X��?��
A�Bm��@   @P  ?���A�(�BiQ�                                    Bx_ѯ4  �          @��?Ǯ@c33?���A�  B���?Ǯ@Z�H?�(�A���B�\                                    Bx_ѽ�  
�          @���?�=q@c33?�G�Ah��B�\?�=q@\(�?���A�G�B���                                    Bx_�̀  
�          @{�?޸R@W
=?��Az=qBz��?޸R@O\)?���A���Bw33                                    Bx_��&             @~�R?��
@Y��?�  Ah��By��?��
@R�\?�=qA�{Bvff                                    Bx_���  
�          @�  ?�\)@XQ�?�ffA���B�Ǯ?�\)@O\)?�\)A��HB}��                                    Bx_��r  �          @���?�G�@Tz�?�p�A��Bxff?�G�@J=q?��A�
=Bs�                                    Bx_�  T          @��R@�@@��?�{A���BL��@�@5?��A��BF{                                    Bx_��  �          @�ff@q�=�\)?��HA��?�=q@q녽�?���A���C�R                                    Bx_�$d  
�          @���@\)�u@	��A�=qC���@\)��=q@Q�A�ffC�3                                    Bx_�3
  T          @�(�@��R�   ?�\)A���C��
@��R�.{?�A��\C�c�                                    Bx_�A�  �          @���@~{=�@��A��
?�(�@~{����@��A��C�O\                                    Bx_�PV  �          @�Q�@}p�    @	��A��
<��
@}p��W
=@��A�RC��                                     Bx_�^�  
�          @���@q�>�=q@(�B�@�Q�@q�<�@p�B��>���                                    Bx_�m�  "          @�=q@~{=���@�RA�\?�(�@~{��@�RA�z�C�*=                                    Bx_�|H  
�          @�=q@z=q>W
=@ffA�p�@AG�@z=q���
@
=A�z�C��{                                    Bx_Ҋ�  T          @���@vff=��
@�A�33?�p�@vff���@�A���C��                                    Bx_ҙ�  �          @�=q@w
=>�p�@=qA�\)@�  @w
=>�@(�B�@                                       Bx_Ҩ:  �          @��@y��?�@Q�A���A(�@y��>���@�A�{@��                                    Bx_Ҷ�  
�          @���@1G�@6ff?���A�=qB7
=@1G�@.{?�(�A��B1�                                    Bx_�ņ  
�          @�?�=q@u��p���P��B��=?�=q@y���\)��G�B�33                                    Bx_��,  T          @���?���@~{�p���K�
B�G�?���@�G������{B��f                                    Bx_���  
�          @�Q�@�
@\)������G�Bz�
@�
@�Q�#�
��\)B{Q�                                    Bx_��x  �          @�G�@33@z=q����=qBn�@33@z=q>��@O\)Bm��                                    Bx_�   "          @�  @{@n�R�����Ba�@{@n�R>u@AG�Ba                                    Bx_��  
Z          @��
@.�R@Y���#�
�   BL��@.�R@X��>�z�@w
=BL�\                                    Bx_�j  
�          @�z�@c�
@�?�{A�G�A�p�@c�
?�{?�ffA��RA���                                    Bx_�,  
\          @�p�@S�
@$z�?��A�
=B�@S�
@�?��A�{B�                                    Bx_�:�  �          @�p�@QG�@(Q�?��A���B\)@QG�@�R?��A�ffB�                                    Bx_�I\  �          @�@E�@7
=?���A���B+�\@E�@.{?�33A��HB%�                                    Bx_�X  
�          @���@2�\@B�\?�p�A�\)B=@2�\@8Q�?�\A�  B7��                                    Bx_�f�  �          @��H@'
=@G�?��RA�\)BH33@'
=@=p�?��A�\)BB33                                    Bx_�uN  �          @�33@/\)@C�
?���A��\B@�\@/\)@:=q?�A���B:�
                                    Bx_Ӄ�  
�          @�(�@(�@`  ?uAIB\ff@(�@X��?��A��BX��                                    Bx_Ӓ�  �          @�=q@%�@P��?�33Axz�BN�@%�@HQ�?�(�A�33BJ                                      Bx_ӡ@  "          @�{@0��@Z=q?�@�33BLG�@0��@U?fffA;\)BI�H                                    Bx_ӯ�  �          @�Q�@=p�@Vff?�@أ�BB  @=p�@Q�?\(�A/\)B?��                                    Bx_Ӿ�  �          @�\)@,(�@dz�=�\)?^�RBS�H@,(�@c33>�
=@�33BS(�                                    Bx_��2  "          @��H@!G�@>{?�(�A�\)BFz�@!G�@5�?�G�A�G�BA33                                    Bx_���  
�          @��@(�@dz�?(�A (�B^Q�@(�@_\)?uAJ=qB\                                      Bx_��~  "          @�G�@��@X��?��AbffB[
=@��@QG�?�\)A�BW(�                                    Bx_��$  T          @�=q@"�\@S�
?��Ac�BRG�@"�\@L(�?���A�\)BN(�                                    Bx_��  �          @���@�@s33>���@�  Bu(�@�@o\)?E�A#
=Bs�                                    Bx_�p  
�          @��?���@�=q    <��
B��
?���@���>���@��HB���                                    Bx_�%  �          @�{@Q�@dz�?���Ae�Ba
=@Q�@\(�?�Q�A��B]=q                                    Bx_�3�  "          @�@
=@W
=?ǮA�  B[��@
=@L(�?��AʸRBU��                                    Bx_�Bb  T          @�
=@$z�@7�@�RA�ffB@ff@$z�@(Q�@   BB6�                                    Bx_�Q  R          @�  ?��R@U�������B���?��R@Tz�>W
=@N{B�                                    Bx_�_�  �          @��>�G�@_\)�2�\��B���>�G�@p  �����B���                                    Bx_�nT  
�          @��\?aG�@r�\�Q�����B��R?aG�@�  ���R��\)B�{                                    Bx_�|�  
�          @���?���@�=q�z���(�B��\?���@�Q��z�����B�{                                    Bx_ԋ�  T          @�Q�?�@�33��(���
=B��q?�@��׿�ff���B�33                                    Bx_ԚF  �          @��?���@����
��Q�B��?���@��H��{���B�L�                                    Bx_Ԩ�  T          @���?�@�녿0����\B��?�@������J=qB�#�                                    Bx_Է�  "          @��?�\)@��׿+��Q�B�{?�\)@�=q����UB��{                                    Bx_��8  �          @���?�=q@xQ�s33�G�
B���?�=q@}p��\)���B��                                     Bx_���  "          @�G�?��@p  �J=q�+
=B|?��@s�
��
=��=qB~=q                                    Bx_��  	`          @�
=@@w��=p����Bv�@@z�H��33����Bx
=                                    Bx_��*  �          @�(�@@qG������p�Bhp�@@xQ�n{�8��Bkp�                                    Bx_� �  R          @�Q�?��@��
?��Aj{B�?��@\)?��A��\B���                                    Bx_�v  
\          @�
=?�  @�{?=p�Ap�B��)?�  @�33?�z�Aq��B�#�                                    Bx_�  �          @��@G�@z�H?&ffA�HBz��@G�@u�?��AX��Bx�H                                    Bx_�,�  
�          @��\@
=q@�Q�>��
@�=qBv��@
=q@|��?8Q�A=qBu�                                    Bx_�;h  
�          @�  ?��@�{>�@�z�B���?��@�(�?aG�A4��B�                                      Bx_�J  T          @��R?�=q@�=q?:�HA33B�p�?�=q@~{?���Amp�B��                                    Bx_�X�  �          @�G�?�  @���?B�\A�B�Q�?�  @�?�
=Ar�HB���                                    Bx_�gZ  	,          @�  ?
=q@��H?Q�A*�HB��?
=q@��?�G�A��B��{                                    Bx_�v   
�          @�����\@~�R?�G�A��B�z��\@u�?�z�A�  B�
=                                    Bx_Մ�  T          @�=q@-p�@^{?0��A�BP=q@-p�@XQ�?��AX��BM\)                                    Bx_ՓL  
(          @�  @Dz�@g�>L��@=qBF��@Dz�@e�?�@�33BE��                                    Bx_ա�  
�          @�\)@<��@l(�=���?�  BM33@<��@j=q>�@���BLQ�                                    Bx_հ�  T          @��@%�@mp�?��@�=qB\�
@%�@hQ�?z�HAEBZz�                                    Bx_տ>  T          @�=q@{@l��?xQ�AE�B`��@{@e�?��A�\)B]z�                                    Bx_���  T          @��
@Q�@Z=q?��
A���B\{@Q�@P��?�\)A��HBWG�                                    Bx_�܊  T          @�33@@dz�?�A|��Bn�\@@[�?��
A��RBj�\                                    Bx_��0  �          @�G�?�@mp�=�\)?h��Bzp�?�@k�>�ff@�z�By��                                    Bx_���  �          @���@p�@aG�?k�AA��B\  @p�@Z=q?��
A�G�BXp�                                    Bx_�|  �          @��
@��@S33?�p�A�\)BX(�@��@HQ�?�A�\)BR\)                                    Bx_�"  
�          @��R@�@S33?�AÅB[\)@�@Fff@
�HA��BTQ�                                    Bx_�%�  �          @��R@
=q@H��@z�A�(�B^G�@
=q@8��@(Q�BffBU{                                    Bx_�4n  T          @�ff?�
=@8Q�@:=qB$��Bn�H?�
=@#�
@L��B8�
Bb�                                    Bx_�C  "          @�z�?�
=@"�\@P  B@=qBrG�?�
=@(�@_\)BT��Bc33                                    Bx_�Q�  
�          @�?�@��@U�BE{BH�?�?�@b�\BVB4p�                                    Bx_�``  
�          @�z�?�ff?���@c�
BX=qB6�
?�ff?��H@n�RBhffB�\                                    Bx_�o  "          @��H?�(�@@7
=B3z�B:z�?�(�?��
@C�
BD=qB(�                                    Bx_�}�  �          @�@�R@e�?L��A'�B]33@�R@^{?�Av{BZ
=                                    Bx_֌R  "          @��@@a�?��
A[�
Ba�@@Y��?�33A��
B^                                      Bx_֚�  
�          @��@(�@W
=?�G�A�
=BW�@(�@Mp�?�{A�BR�H                                    Bx_֩�  "          @��@�\@Tz�?\A�
=B]��@�\@H��?�{Ȁ\BW��                                    Bx_ָD  �          @��\@\)@S�
?�z�A��\B_��@\)@G�@   A�Q�BYQ�                                    Bx_���  �          @���@�
@Y��?�\)A��HBk
=@�
@Mp�?�(�A�{Be�                                    Bx_�Ր  �          @��@G�@U?��A�p�Bkff@G�@HQ�@��A�RBd                                    Bx_��6  
Z          @�(�@p�@U?�\A��RBa�@p�@HQ�@
=A���B[{                                    Bx_���  "          @���@ff@Tz�?�p�A���Bf�\@ff@G
=@�A�B_�                                    Bx_��  
�          @�(�@z�@R�\?޸RA��B[G�@z�@E�@�A��BT=q                                    Bx_�(  �          @�Q�@�@J�H?�=qA��BT�@�@?\)?�z�A���BN33                                    Bx_��  �          @�p�?��H@`��?��\Ac\)BsQ�?��H@XQ�?��A�z�Bo��                                    Bx_�-t  "          @�\)@�@e?z�@��RBn  @�@`��?uAT  Bk��                                    Bx_�<  "          @�{@�R@^{>��@�Q�BeG�@�R@Y��?W
=A;33Bc(�                                    Bx_�J�  �          @��R@{@tz�>k�@?\)Bo�@{@q�?#�
Ap�BnG�                                    Bx_�Yf  
�          @��@z�@l�Ϳz����Bs
=@z�@o\)�.{�z�Bt�                                    Bx_�h  �          @�Q�@�
@N{�#�
���Be�\@�
@N{>B�\@6ffBez�                                    Bx_�v�  	�          @x��?��
@N{���
��(�B�?��
@W
=������B���                                    Bx_ׅX  T          @���?�ff@g����\��33B���?�ff@n�R�^�R�B=qB��                                    Bx_ד�  T          @��?˅@g������y��B��\?˅@n{�333���B��                                    Bx_ע�  
�          @��?Ǯ@e���R���B��
?Ǯ@l�ͿW
=�=�B�(�                                    Bx_ױJ  
�          @�ff?�\)@dz῾�R����B��?�\)@mp�����qp�B�Ǯ                                    Bx_׿�  �          @�33��@���8���8p�B�uÿ�@,���'
=�"ffB��                                    Bx_�Ζ  f          @��\��ff@*=q�9���5�
B�=q��ff@=p��%��B�(�                                    Bx_��<  "          @u��?����{�<z�C�)��?�=q���*ffCk�                                    Bx_���  T          @w����0���7
=�F�CD������p��:�H�L33C={                                    Bx_���  �          @|���*�H��=q��H�  CR�=�*�H��ff�%��${CM�f                                    Bx_�	.  �          @���,(�>���?\)�?C0���,(�>��=p��<�
C)�3                                    Bx_��  �          @����6ff<��'��*�C3O\�6ff>����&ff�(�C-z�                                    Bx_�&z  "          @���'
=>�ff�C33�D{C*8R�'
=?J=q�>�R�>(�C#0�                                    Bx_�5   "          @����
=����O\)�W\)C::��
==�G��P  �X33C1^�                                    Bx_�C�  T          @����(��#�
�K��R  C4:��(�>�33�J=q�P33C+��                                    Bx_�Rl  �          @\)�(������J=q�Q  C6:��(�>���I���P�C-��                                    Bx_�a  "          @������#�
�I���P�CC  ����\)�L���U  C:�H                                    Bx_�o�  T          @�=q�%�fff�;��;��CG��%�\)�@���B�C@.                                    Bx_�~^  �          @�33��      �l��G�C3����  >�
=�j�H�p�C&n                                    Bx_؍  �          @��H�޸R<#�
�l(���C3��޸R>�G��j�H��HC%�                                    Bx_؛�  �          @����{=�G��L���Qp�C1\)�{>��J�H�NG�C)�                                    Bx_تP  T          @��H�
==�G��^{�j�RC1��
=?�\�[��f�C&��                                    Bx_ظ�  �          @���
=q>�\)�^{�gQ�C,���
=q?.{�Z�H�a��C"��                                    Bx_�ǜ  T          @����{>#�
�_\)�f  C/��{?��\���a�RC%�3                                    Bx_��B  "          @����!G��B�\�R�\�Q�RC8:��!G�>B�\�R�\�Q�C/�{                                    Bx_���  �          @�33� �׿z��H���JffC@�3� �׾W
=�K��N�C8�                                    Bx_��  �          @�(��'
=�����J�H�I��C:���'
==�\)�K��J��C2��                                    Bx_�4  
�          @�{�ff�u�Z�H�]�\C9��ff>#�
�[��^{C0.                                    Bx_��  �          @�����u�W
=�XffC5z���>����U�V�C,B�                                    Bx_��  
�          @|���0�׾�z��,���0��C:��0��<��
�.{�1��C3}q                                    Bx_�.&  
�          @}p���
�}p��A��I  CK!H��
�!G��G��Q�HCC!H                                    Bx_�<�  
�          @\)�Q쿸Q��@  �E�
CV#��Q쿊=q�I���S��CN޸                                    Bx_�Kr  
�          @|(��
=?�{�#�
�,��C\�
=?�33�Q��  C�                                    Bx_�Z  �          @|���@ ����H�ffCff�@��
=q���C�                                    Bx_�h�  �          @������?���2�\�1{CQ����?�{�%��!G�C:�                                    Bx_�wd  "          @���Q�?��\�L���P��C(��Q�?���AG��Az�CQ�                                    Bx_ن
  
�          @�=q���?���G
=�H=qC�=���?�  �:�H�8Q�C��                                    Bx_ٔ�  T          @��H�z�?���5��/�
CG��z�@��%�{C	��                                    Bx_٣V  T          @�=q�(�?����AG��@(�C��(�?����333�.��C
aH                                    Bx_ٱ�  T          @�����?�Q��?\)�?p�C�\��?�ff�5��1��C�                                     Bx_���  "          @�Q��   ?���:�H�;�C��   ?�
=�1G��/p�C33                                    Bx_��H  T          @|(��"�\?Q��5��:�C"��"�\?�z��-p��0Cs3                                    Bx_���  
�          @|�Ϳ��H?\)�W
=�j��C$Ϳ��H?xQ��P���`�C��                                    Bx_��  
Z          @y���"�\?Y���.�R�6(�C!�{�"�\?��&ff�+��C5�                                    Bx_��:  "          @{��>{?����z���C���>{?����
�H�  C�                                    Bx_�	�  T          @x���7�?}p��=q�
=C ���7�?��
�G����C�                                    Bx_��  �          @tz��G�?&ff����HC(B��G�?fff���R��{C#�f                                    Bx_�',  T          @tz��>�R?�\�����C*@ �>�R?J=q���{C%0�                                    Bx_�5�  �          @z=q�Fff?&ff���
=C(!H�Fff?n{���  C#L�                                    Bx_�Dx  
�          @xQ��S33?
=��Q���Q�C)޸�S33?Tz������Q�C%�                                    Bx_�S  �          @x���U�>�Q��(���C-�H�U�?����z���\)C)�=                                    Bx_�a�  �          @{��\(�?=p���  ��\)C'��\(�?s33����ř�C$�
                                    Bx_�pj  
�          @���[�?p�׿��H��=qC$���[�?�
=������G�C!�                                    Bx_�  
�          @��R�j=q?aG���\)��
=C&��j=q?�{�޸R���
C#.                                    Bx_ڍ�  
�          @�z��i��?xQ���H��z�C%)�i��?�
=������{C"�                                    Bx_ڜ\  T          @}p��^{?��ÿ������C"���^{?�G���
=���HC��                                    Bx_ګ  "          @����^{?}p����
�љ�C$��^{?��H������C �{                                    Bx_ڹ�  T          @�Q��a�?�����p���33C���a�?��Ϳ���o�
C�f                                    Bx_��N  
�          @�Q��c33?��
��=q��33C 33�c33?�Q쿔z���(�C�3                                    Bx_���  "          @~{�c33?��R��(����
C ���c33?�녿�ff�v�\C�H                                    Bx_��  �          @�Q��`��?�
=�u�]�Cff�`��?���:�H�)G�C��                                    Bx_��@  "          @}p��W
=?��R����CJ=�W
=@�\����uC��                                    Bx_��  l          @���?\)@�ÿ��
��(�Ch��?\)@�\��  �mG�C��                                    Bx_��  �          @���9��@G��ff��\)C!H�9��@녿����ѮC޸                                    Bx_� 2  �          @��H�%�?����   ���C���%�@���{���C
Ǯ                                    Bx_�.�  �          @{��
=q?�33�333�7(�C���
=q@ ���#�
�#�RC	+�                                    Bx_�=~  �          @vff�(�?�ff�Q��G�C�H�(�@ff���ffC^�                                    Bx_�L$  �          @�(��4z�@��\)�=qC0��4z�@�
������p�C�f                                    Bx_�Z�  T          @}p��,��?�\)����33C=q�,��@�ÿ�\)��{C��                                    Bx_�ip  �          @��R�]p�?ٙ���
=��z�Cٚ�]p�?�zῸQ���=qC#�                                    Bx_�x  �          @��g
=?�
=������C^��g
=?�\)�����G�CǮ                                    Bx_ۆ�  �          @�{�vff?�=q�s33�RffC �q�vff?�Q�E��(��CxR                                    Bx_ەb  �          @�{�vff?��Ϳh���H��C ���vff?��H�8Q��ffC:�                                    Bx_ۤ  �          @����vff?�
=�s33�S�
C"�3�vff?�ff�G��-�C!aH                                    Bx_۲�  �          @���u�?}p����\��  C%z��u�?�33��\)�z�\C#:�                                    Bx_��T  �          @����vff?c�
��G���\)C'
=�vff?�ff�����|Q�C$�                                    Bx_���  �          @�p��\)?\(��Y���<��C'�)�\)?xQ�:�H�!�C&aH                                    Bx_�ޠ  �          @�ff���H?L�Ϳ�\��p�C(����H?\(��Ǯ���C()                                    Bx_��F  �          @�p��z=q?�z�B�\�)��C#u��z=q?�G�����Q�C"5�                                    Bx_���  �          @��~�R?�\)�z���p�C$B��~�R?�Q��
=���RC#Y�                                    Bx_�
�  T          @���}p�?�����{C#���}p�?�p���33��(�C"��                                    Bx_�8  �          @���{�?����\��z�C$s3�{�?�33��33��ffC#��                                    Bx_�'�  �          @�p�����?u�������C&�=����?��\��{���
C%Ǯ                                    Bx_�6�  �          @�z��\)?�G�������C%�\�\)?����=q�o\)C%+�                                    Bx_�E*  �          @�z��~{?����������C$�)�~{?�녾u�W�C$                                      Bx_�S�  �          @�z��|��?�����������C##��|��?�p��#�
�	��C"��                                    Bx_�bv  �          @�33�xQ�?�������\)C"�{�xQ�?�G���{����C"                                    Bx_�q  �          @���o\)?�=q�G��2{C p��o\)?�
=�
=��C!H                                    Bx_��  �          @���n{?�(��333��
Cu��n{?�ff���H��p�CT{                                    Bx_܎h  �          @��\�`��?���u�[�
C���`��?�z�333��HCxR                                    Bx_ܝ  �          @��\�e?\��=q�xQ�C)�e?�z�\(��C\)C@                                     Bx_ܫ�  T          @�=q�b�\?�Q쿮{��
=Cٚ�b�\?�\)������HCc�                                    Bx_ܺZ  �          @�Q��k�?�ff�c�
�MG�C �{�k�?�z�0����C                                    Bx_��   �          @�33�j=q?�  ���\�f�HC���j=q?У׿J=q�2�RC�R                                    Bx_�צ  �          @�=q�g�?��
�����t(�C
=�g�?��W
=�>=qC5�                                    Bx_��L  �          @�\)�p  ?�G��Tz��4��C�)�p  ?�{�\)���C��                                    Bx_���  �          @����mp�?�
=�n{�H  C�=�mp�@�\�#�
���C+�                                    Bx_��  �          @���q�?�\)�����k
=C�)�q�@ �׿Q��-p�C�                                    Bx_�>  �          @�=q�e@p��xQ��P  Ch��e@z�#�
�  C�                                    Bx_� �  �          @��Vff@\)����s\)C8R�Vff@Q�@  �&�\C��                                    Bx_�/�  T          @�{�W
=?�p���(���p�C���W
=@�����C
                                    Bx_�>0  �          @�ff�U�?�  �33��RC�3�U�?�����̏\C��                                    Bx_�L�  �          @��R�\?��H���ڸRC���R�\?�p������\)C�                                    Bx_�[|  �          @����Mp�?xQ��(��p�C#:��Mp�?����G���C��                                    Bx_�j"  �          @��H�Z�H?�Q����ң�C ٚ�Z�H?�Q�������C!H                                    Bx_�x�  T          @��
�W�?�33��������C�)�W�?�׿�=q����C��                                    Bx_݇n  �          @�(��s33>�׿�G���ffC,���s33?333��
=���
C)��                                    Bx_ݖ  �          @��H�{�>u�����s\)C0���{�>��Ϳ��\�h(�C.:�                                    Bx_ݤ�  �          @�33�vff�#�
��z�����C4
�vff>W
=�������C0��                                    Bx_ݳ`  �          @��|(��\)��=q���RC6��|(�=u��=q��p�C3
                                    Bx_��  �          @�\)�~{>B�\��
=���\C1J=�~{>��Ϳ����G�C.0�                                    Bx_�Ь  �          @����  <��
��\)��p�C3����  >k��������C0��                                    Bx_��R  �          @������H=#�
������ffC3h����H>k���
=��{C0ٚ                                    Bx_���  �          @�G�������Ϳ�{�o�C9�
����k���33�z�\C7(�                                    Bx_���  �          @�����H���Ϳ�G��]p�C9�����H�u����hz�C7O\                                    Bx_�D  �          @�����ff<#�
�G��(��C3޸��ff>��E��&=qC28R                                    Bx_��  �          @�Q���ff>L�Ϳ!G����C18R��ff>����
=� (�C/�                                    Bx_�(�  #          @�Q����?E��E��'�C)W
���?aG��#�
�
=C'ٚ                                    Bx_�76  �          @��H�|��?����\)C,���|��?�����H��{C+aH                                    Bx_�E�  �          @�  ��@N{�������B��)��@XQ�J=q�3
=B�B�                                    Bx_�T�  �          @�Q��N�R@�
������RC� �N�R@\)��  �d��C5�                                    Bx_�c(  �          @����:=q@%�������C
n�:=q@0�׿c�
�G�C��                                    Bx_�q�  �          @�=q�5@'
=���H��33C	T{�5@1G��G��1��C��                                    Bx_ހt  �          @��H�!G�@7
=����33Cc��!G�@C33�p���V�\C��                                    Bx_ޏ  �          @�Q��*=q@)����p����RC{�*=q@3�
�J=q�7�CaH                                    Bx_ޝ�  �          @�  �Q�@E��ff��ffB�.�Q�@P�׿G��5�B�\)                                    Bx_ެf  �          @z=q��{@S33��z����RB����{@^�R�\(��TQ�B�B�                                    Bx_޻  �          @|��>���@`  ����
=B�Q�>���@n�R��=q����B��f                                    Bx_�ɲ  �          @vff��G�@Z�H��
=�ϙ�B\��G�@h�ÿ�����\)B���                                    Bx_��X  �          @z=q�k�@?\)�   ��B�녾k�@W
=��p���p�B��                                    Bx_���  �          @��
�:=q@"�\��������C
���:=q@/\)���\�d��C                                    Bx_���  �          @�G��1G�@/\)��  ��(�CJ=�1G�@?\)���
���
C�=                                    Bx_�J  �          @�G��>{@1G���\)��G�C�3�>{@=p��fff�AG�C{                                    Bx_��  �          @��
�3�
@+�������G�CaH�3�
@7
=�c�
�G�Cs3                                    Bx_�!�  �          @�G��/\)@{��=q��
=C	��/\)@,�Ϳ������Ck�                                    Bx_�0<  �          @\)�6ff?�\)��p���C���6ff@Q쿳33���C0�                                    Bx_�>�  �          @vff�R�\?��Ϳ����\)C�R�\?�G��G��<��C��                                    Bx_�M�  �          @~�R�:=q@p������zffC���:=q@&ff�#�
��C
.                                    Bx_�\.  �          @�  �G�@L�Ϳ.{�Q�B��H�G�@P�׾�����B��q                                    Bx_�j�  �          @z=q��ff@dz����=qB��쿆ff@fff>��@\)BԔ{                                    Bx_�yz  �          @��
>��@���=���?�\)B���>��@|��?G�A0��B�aH                                    Bx_߈   �          @z�H��@w
=>B�\@2�\B���@qG�?W
=AG33B�Q�                                    Bx_ߖ�  �          @���>�33@��?n{AK�B�#�>�33@y��?�\)A��B��{                                    Bx_ߥl  �          @�Q�?�\@��?�RA	G�B��?�\@|(�?���A��RB�Q�                                    Bx_ߴ  T          @�������@�Q�>W
=@:�HB�p�����@z�H?c�
AL(�B��                                    Bx_�¸  �          @�Q�>aG�@�\)>�=q@l(�B��>aG�@��
?�  AYp�B�                                    Bx_��^  �          @�ff?�@��
>�@�G�B�33?�@~�R?�z�A�
=B���                                    Bx_��  �          @�ff?aG�@���?0��Az�B�k�?aG�@u?���A�(�B�L�                                    Bx_��  �          @�p��#�
@�z�>�
=@�\)B�z�#�
@�  ?���A{�B���                                    Bx_��P  �          @�(�<#�
@�33>.{@z�B���<#�
@�Q�?aG�AE�B���                                    Bx_��  �          @�
==L��@�ff=��
?�z�B���=L��@��
?Q�A3�
B���                                    Bx_��  �          @���#�
@���&ff�\)B�#׼#�
@�G�=���?�33B�#�                                    Bx_�)B  �          @�=q>�p�@��?��RA�=qB��{>�p�@u�?���A�
=B���                                    Bx_�7�  �          @���>��
@z�H?�=qA�33B���>��
@e�@  B 33B�                                    Bx_�F�  �          @�Q�>��@^{@�HBQ�B�p�>��@?\)@@  B4
=B�\                                    Bx_�U4  �          @{����
@"�\�5��@ffB�논��
@@  ���ffB�Ǯ                                    Bx_�c�  �          @tz�>�@��=p��J��B�Q�>�@7
=��R�"(�B��                                    Bx_�r�  �          @n{?8Q�@3�
�33�=qB�33?8Q�@K���  ��=qB�ff                                    Bx_��&  �          @j�H?��\@9�����R��RB�#�?��\@Mp���Q����HB���                                    Bx_���  �          @�G�?��@Vff������B�{?��@i�������=qB��
                                    Bx_��r  �          @n{?��@Z=q��{���B�(�?��@c�
�����
=B��\                                    Bx_�  �          @e�?��@_\)�(����B�#�?��@b�\<�?
=qB�ff                                    Bx_໾  �          @�G�?�(�@xQ쿜(���=qB��?�(�@��ÿ   �׮B�ff                                    Bx_��d  �          @�ff?��H@s�
��=q�c
=Bz�
?��H@|(���p���G�B~
=                                    Bx_��
  T          @���@(�@p�׾�����
=Bo33@(�@qG�>���@z�HBop�                                    Bx_��  �          @���@�@�Q��\��  Bz�@�@���>k�@:�HBz�R                                    Bx_��V  �          @�Q�@(�@z�H����(�Bs33@(�@|(�>�  @J�HBs�R                                    Bx_��  �          @�ff@��@�(��Ǯ��\)Bw��@��@�(�>Ǯ@�ffBw��                                    Bx_��  �          @�Q�@��@��H��ff���RBn33@��@�33>��R@q�Bn�                                    Bx_�"H  �          @�{@Q�@�  ���H���RBl�H@Q�@���>�=q@R�\Bmff                                    Bx_�0�  �          @�ff@!�@z=q��=q�P��Bd��@!�@y��>��@���Bd{                                    Bx_�?�  �          @���@��@��>��R@n�RB|��@��@��?���AU�Bz                                      Bx_�N:  �          @���@�@��>k�@5�B}@�@�G�?�G�AHQ�B{�                                    Bx_�\�  �          @��
?��H@��W
=�)��B���?��H@�z�?z�@�RB�G�                                    Bx_�k�  �          @���?�
=@���=�Q�?���B�\?�
=@�?fffA3�B�.                                    Bx_�z,  �          @�Q�@
=@�\)>�ff@��RB~�@
=@�=q?�  At  Bzz�                                    Bx_��  T          @�Q�@:�H@mp�>L��@=qBOG�@:�H@fff?fffA0Q�BL
=                                    Bx_�x  �          @��\@.{@~{>�\)@UB^Q�@.{@u?��
AF�\BZ��                                    Bx_�  �          @��@Vff@\��>�
=@�\)B7p�@Vff@S�
?���AMB2�\                                    Bx_��  �          @���@@  @r�\>\)?У�BNQ�@@  @l(�?\(�A$��BKff                                    Bx_��j  �          @�p�?�Q�@�\)�\��
=B���?�Q�@�
=>�G�@��B��=                                    Bx_��  �          @��@�@|(��!G���=qBm@�@\)>��?�=qBn�                                    Bx_��  T          @��@%�@mp��\)��=qB\�@%�@p  >.{@
�HB]                                    Bx_��\  �          @��@E�@S33�.{�(�B;��@E�@W����
�k�B>G�                                    Bx_��  �          @��H@H��@R�\����Q�B9z�@H��@Tz�>L��@��B:ff                                    Bx_��  �          @�G�@��@p�׿�����Bc�@��@s33>��?�\)Bd�                                    Bx_�N  �          @�ff?�33@z=q���
��Q�B�8R?�33@y��>�ff@�
=B�{                                    Bx_�)�  �          @�@QG�@!�?�ffA�B�@QG�@{?��
A��B��                                    Bx_�8�  �          @�z�@g
=@z�?G�A$Q�B�@g
=@�?�(�A�=qA�\)                                    Bx_�G@  �          @���@H��@*�H?��AfffB!�R@H��@=q?���A��B
=                                    Bx_�U�  �          @��@:�H@<(�?���As�
B4��@:�H@*=q?�Q�A�B)(�                                    Bx_�d�  �          @��
@E@:�H?z�HAO�B-ff@E@*=q?��A��B"��                                    Bx_�s2  �          @�
=@`��@*=q?B�\A��B�\@`��@��?��
A��B�R                                    Bx_��  �          @�{@x��@�\>�Q�@�{A�33@x��?�?B�\A�
A�z�                                    Bx_�~  �          @�Q�?�\)@`  ?��HA�\)B�33?�\)@E@Q�B��B�                                    Bx_�$  �          @�  ?�p�@R�\?�Q�A�=qBx�
?�p�@5�@#�
B�Bi�                                    Bx_��  �          @�{?��@O\)@33B\)B��=?��@-p�@:=qB0z�B���                                    Bx_�p  �          @��?\(�@A�@-p�B"�B�aH?\(�@=q@QG�BO�B���                                    Bx_��  �          @�z�@�@8��?���A癚B[�@�@�@   BffBHff                                    Bx_�ټ  �          @��@L��@#�
���
�aG�B��@L��@ ��?�\@�p�B�                                    Bx_��b  �          @\)@8Q�@ ��>�  @�G�BG�@8Q�?�z�?(��A,  B�                                    Bx_��  �          @_\)?�ff@)���L���n{Bn�H?�ff@(��>�33@�33BnG�                                    Bx_��  �          @p  @{@>�R��Q���G�BU33@{@?\)>�=q@��HBU�\                                    Bx_�T  �          @vff@G�@J=q?   @��HBez�@G�@>�R?�33A��B_�                                    Bx_�"�  �          @�  @	��@U�>8Q�@%Bdz�@	��@N{?^�RAJ�HB`�                                    Bx_�1�  �          @~{?�
=@]p�=u?aG�Bs�?�
=@W�?G�A6�HBp��                                    Bx_�@F  T          @{�?�\@`��>k�@Q�B}33?�\@XQ�?uAap�By��                                    Bx_�N�  �          @tz�?\@_\)>#�
@�RB���?\@W�?fffAY�B���                                    Bx_�]�  �          @|(�?�  @`  �\)�z�B}�?�  @\��?��A��B|\)                                    Bx_�l8  �          @tz�?\@Z�H��������B�{?\@Y��>�G�@�{B��H                                    Bx_�z�  �          @\��?�ff@G�?��A z�B��q?�ff@:�H?�  A���B�                                      Bx_㉄  �          @0��?���@(�>�z�@�  Bff?���@z�?L��A�=qBz�H                                    Bx_�*  �          ?�=q?�?\(�>\)A   Bhz�?�?O\)>��
A�p�Bb(�                                    Bx_��  �          ?�33=�?c�
�:�H�B�.=�?��
������B�#�                                    Bx_�v  �          ?�녽�Q�?�G��+���B�{��Q�?��׾�(���(�B�                                    Bx_��  �          @�
?333?��R?�Q�B�
B���?333?�Q�?�p�B<(�Bo                                      Bx_���  �          @
=?���?�Q�>��HAZffBc�?���?��?W
=A��BYff                                    Bx_��h  �          @33?L��?�G�?\(�A�{Bw�?L��?��?�B��Bh
=                                    Bx_��  �          @ ��?�R?�Q�?��HB33B���?�R?��?�  BD�BvQ�                                    Bx_���  �          @Q�?&ff?��R?�\)B \)B��
?&ff?�33?�z�BM�RBrp�                                    Bx_�Z  �          @�\?�?\?��HB\)B��3?�?�(�?�G�BB�B�#�                                    Bx_�   �          ?�=q>��?�(�?�ffB�
B�G�>��?���?���B>��B�                                      Bx_�*�  �          ?�  �L��?�
=?z�HB	z�B��L��?�
=?��\B<z�B�Ǯ                                    Bx_�9L  �          ?��<�?�
=?�p�B"�
B�z�<�?�\)?\BV
=B�Ǯ                                    Bx_�G�  �          ?��
=�G�?�(�?��
B9(�B��=�G�?fff?\BlG�B�u�                                    Bx_�V�  �          ?�\)>8Q�?E�?���Bp��B���>8Q�>�G�?\B��B��R                                    Bx_�e>  �          ?��þ��?E�?�ffBl(�B��þ��>�?�Q�B�#�B��                                    Bx_�s�  �          @G��.{?���?��BJ�\B��)�.{?0��?޸RBs�CT{                                    Bx_䂊  �          ?��H�8Q�?��?�Q�B=�\BøR�8Q�?s33?�Q�Bp�Bɀ                                     Bx_�0  �          ?�\��G�?��H?�z�B.�B��ͽ�G�?h��?�33BbB�8R                                    Bx_��  �          ?�(�?#�
?333?�\)B]=qB=
=?#�
>�p�?��RB~G�A�=q                                    Bx_�|  �          ?���>���Q�?�  B��fC�.>����H?�
=B�p�C��H                                    Bx_�"  �          ?�p�>�=�Q�?�{B��A0��>��#�
?���B�aHC��                                    Bx_���  �          ?�ff>�z�#�
?}p�B��C�l�>�zᾅ�?uB��\C���                                    Bx_��n  �          ?�
==#�
��\)?�=qB��HC�U�=#�
��p�?��
B�L�C�k�                                    Bx_��  �          ?��þ#�
��\?��B��C|�þ#�
�B�\?}p�BO�\C�3                                    Bx_���  �          ?����Q쿇�?+�B �RC�����Q쿘Q�>��A�G�C���                                    Bx_�`  �          ?�G�>.{��
=�(���\��C�  >.{����=p�8RC��=                                    Bx_�  �          ?�(�?�R?�ff�5��
=B��)?�R?�
=�Ǯ�^�HB��                                    Bx_�#�  �          @�>�ff?�(�?J=qA���B�
=>�ff?�p�?�p�B
  B���                                    Bx_�2R  �          @�>\?���?���B'B��R>\?�Q�?�\B[�RB�p�                                    Bx_�@�  �          @G�>��
?�(�?��B"=qB�.>��
?���?˅BV�RB�                                      Bx_�O�  �          @zἣ�
@   ?���A�\)B�����
?�z�?У�B1�B�W
                                    Bx_�^D  �          @녾��?��H?�=qA��
B��H���?�33?\B%��BӮ                                    Bx_�l�  �          @�Ϳ
=q?�z�?�\)B
33Bӳ3�
=q?��?��
B>=qB��                                    Bx_�{�  T          @DzῈ��@��?ǮA�{B�\����?��H@�B,(�B�8R                                    Bx_�6  �          @9����p�@z�?�B�B�녾�p�?�{@
�HBCB�G�                                    Bx_��  �          @U��#�
@(��?�z�B�HB�W
�#�
@ff@\)BB  B��H                                    Bx_姂  �          @Q녿u@@�B��B��\�u?��@!�BL�
B�aH                                    Bx_�(  �          @<���(�?L��?�
=B  C���(�>�p�?���BQ�C*T{                                    Bx_���  �          @-p����
�L��?�z�B!CLaH���
��
=?�B��CU��                                    Bx_��t  �          @H����<��
@�B+�C3}q�녾��H@z�B&=qC@+�                                    Bx_��  �          @W
=�(�<��
@�
B-�
C3��(��
=q@  B'��C@n                                    Bx_���  �          @\(��'
=�\)@�RB!��C7+��'
=�.{@Q�B�HCB��                                    Bx_��f  �          @`  �0  ���R@Q�B=qC:���0  �Q�?��RB
�HCD��                                    Bx_�  �          @dz��	������@(�B/�CT��	����33@�\BC]xR                                    Bx_��  �          @o\)��ÿǮ@)��B4{CX�����	��@(�B
=Ca33                                    Bx_�+X  �          @n{���޸R@)��B6z�C^5ÿ��@��B33Cf��                                    Bx_�9�  T          @xQ���R�   @*�HB-�
Ca
���R�%@ffB�HChxR                                    Bx_�H�  �          @l�Ϳ�����@#33B.  Ch� ����,��?���A���Cn��                                    Bx_�WJ  �          @l(���ff�Q�@$z�B1(�Ci�3��ff�,(�?��HB
=Cp�                                    Bx_�e�  �          @g�����ff@"�\B2ffC`E����@G�Bz�Ch�                                    Bx_�t�  �          @`  �����Q�@\)B5ffC_Q����  ?��RB
�HCg�=                                    Bx_�<  �          @hQ��33��
=@&ffB6C]Y���33�G�@ffBG�Cf�                                    Bx_��  �          @mp������@!G�B+(�CX�����p�@�B�HCas3                                    Bx_栈  �          @j�H�
�H��33@(�B'
=CYE�
�H�p�?�Q�B z�Ca�                                     Bx_�.  �          @h�ÿ��R��\)@%B5  C[J=���R�{@ffB�\Cd8R                                    Bx_��  �          @dz��
�H�Ǯ@z�B#��CW��
�H�?���A���C_��                                    Bx_��z  �          @e���@��B�C\�����
?�A�Cc�\                                    Bx_��   �          @X�����
=?�\A��HCW�f��z�?��A�\)C]��                                    Bx_���  �          @i���'���?��A�{CX(��'��G�?���A��\C\޸                                    Bx_��l  T          @a��!G����?˅A�
=CX�!G��{?��A�Q�C]u�                                    Bx_�  �          @XQ��%���?���A��CW���%��?0��A;�CZ�q                                    Bx_��  �          @QG�� �׿��H?�
=A�z�CV0�� �׿���?5AH��CY�)                                    Bx_�$^  �          @W
=�"�\����?Q�Aj=qCW���"�\��(�>��
@�\)CY�R                                    Bx_�3  �          @W
=�#�
�?��A�HC[B��#�
�
=q�#�
��C\0�                                    Bx_�A�  �          @N�R�p���p�?��A+\)CZ�{�p��z�=�\)?�Q�C\
=                                    Bx_�PP  �          @O\)�����?G�A`��C_����G�>L��@g
=C`�\                                    Bx_�^�  �          @<�����޸R?uA�z�CZ�{����
=>�A��C]��                                    Bx_�m�  �          @<(���33��p�?z�HA�z�Cb=q��33�
�H>�(�A�RCd�=                                    Bx_�|B  �          @.{�
=���
?:�HAz�HCW��
=��z�>��R@���CZQ�                                    Bx_��  �          @(Q���H��\)?�G�A��HC_J=���H����?��AECb�q                                    Bx_癎  �          @I���
=���H?�(�A�(�C^�R�
=�p�?+�AD��Cbn                                    Bx_�4  �          @s33�Tz�G���������CA0��Tz᾽p���{��C:T{                                    Bx_��  �          @�33�c�
��(���p����C:�)�c�
=�G��G����C2E                                    Bx_�ŀ  �          @�Q��^{�W
=��ff���CA���^{���ÿ��H��Q�C9z�                                    Bx_��&  �          @�  �a녿aG���33�£�CA��a녾�녿����ظRC:�)                                    Bx_���  �          @����g
=�E���{��=qC@\�g
=���
��  �θRC9                                    Bx_��r  �          @�Q��fff�^�R���
��  CA���fff��(����H��  C:Ǯ                                    Bx_�   �          @|���e���
��G����HCD�e�(�ÿ��R��\)C>aH                                    Bx_��  �          @|���dzῨ�ÿ���tz�CHE�dz�}p���{����CC��                                    Bx_�d  �          @^{�C33���R�+��3�
CM���C33��  ��ff��
=CJG�                                    Bx_�,
  �          @j=q�L(��\�p���o\)CMff�L(����H�������HCH��                                    Bx_�:�  �          @~�R�\�Ϳ��H��G��j�RCNc��\�Ϳ��׿�
=���RCI                                    Bx_�IV  �          @\)�^�R���R��  ��ffCK(��^�R���Ϳ�{��=qCEu�                                    Bx_�W�  �          @\)�\(����ÿ��
���CL��\(���z��33��CF��                                    Bx_�f�  �          @~�R�^�R��p����R���
CK!H�^�R����������
CEk�                                    Bx_�uH  �          @{��Z�H���R��p���ffCK� �Z�H����˅��33CE�                                     Bx_��  �          @����_\)��Q쿳33���\CJY��_\)��  �޸R��z�CC�R                                    Bx_蒔  �          @~{�aG���G����\�m��CK0��aG��������(�CFk�                                    Bx_�:  �          @w��a녿�z�5�)�CI��a녿�zῈ������CF0�                                    Bx_��  �          @z=q�`�׿�\)����(�CL�R�`�׿�녿��\�qG�CI��                                    Bx_辆  �          @|���a녿�
=���� ��CMh��a녿��H�}p��ip�CJu�                                    Bx_��,  �          @\)�e���33�z����CL���e���
=��G��j�RCI��                                    Bx_���  �          @~�R�c33��  �Ǯ���HCNG��c33��=q�\(��G\)CK�3                                    Bx_��x  �          @|���^�R���ͽ���G�CO���^�R�޸R�!G��
=CN�{                                    Bx_��  �          @}p��\(���(��#�
�z�CQǮ�\(���׿
=�Q�CP�f                                    Bx_��  �          @j�H�H�ÿ��<��
>��RCR�
�H�ÿ��   ��CQ��                                    Bx_�j  �          @aG��333�>�z�@��RCX�333����R���CX�R                                    Bx_�%  �          @g��8���
�H>L��@N{CX���8����þ����=qCX�                                    Bx_�3�  �          @e�6ff�	��>��R@���CY\�6ff�	���������CY{                                    Bx_�B\  �          @c33�/\)���>u@|��CZٚ�/\)���Ǯ���HCZ�{                                    Bx_�Q  �          @j=q�5�33>k�@j=qC[��5�G���(���{CZ��                                    Bx_�_�  �          @i���B�\� ��>�@z�CU���B�\��(���G��߮CT�                                    Bx_�nN  �          @j=q�Dz��p�>aG�@Y��CTٚ�Dz���H��Q����CT�{                                    Bx_�|�  �          @k��G
=����<�>��CT��G
=��\)����CS
                                    Bx_鋚  �          @j�H�I����=�G�?��
CRQ��I����ff��
=���CQ�R                                    Bx_�@  �          @n�R�G
=�33>\)@�CUc��G
=�   ����G�CTǮ                                    Bx_��  �          @mp��G
=� ��>\)@ffCT�)�G
=���H��ff��p�CTE                                    Bx_鷌  �          @mp��?\)�ff?(�A��CV�q�?\)���#�
�W
=CX)                                    Bx_��2  T          @p  �E��>���@���CV��E����z�����CV33                                    Bx_���  �          @mp��Fff� �׾�����{CT��Fff��ff�u�pz�CR:�                                    Bx_��~  �          @p���C�
��(��h���_�CT��C�
��\)��Q���Q�CO�3                                    Bx_��$  �          @qG��@  ����(���\)CT�f�@  ���R��p���=qCNT{                                    Bx_� �  �          @qG��<�Ϳ��H������CU�)�<�Ϳ�  ��ff���CN�R                                    Bx_�p  �          @n�R�AG���p��h���b�HCU@ �AG��У׿�����\)CPT{                                    Bx_�  �          @k��AG���33�L���K33CT33�AG���=q�������
CO�                                    Bx_�,�  �          @i���1G����\�ŮCU+��1G�������R�(�CL��                                    Bx_�;b  �          @s�
�6ff��\)��Q���G�CU5��6ff����
�H�
CLL�                                    Bx_�J  �          @u��0�׿�׿�{��G�CVT{�0�׿�  ���CLh�                                    Bx_�X�  �          @g���H��
=�����CZ����H����33��CPff                                    Bx_�gT  �          @w
=��ÿ�G��{���CX}q��ÿu�8Q��@��CIٚ                                    Bx_�u�  �          @o\)�{��(�����Q�CR���{�333�.�R�:��CC�q                                    Bx_ꄠ  �          @s33�E���������\CR�=�E��Q�����ŮCL�q                                    Bx_�F  �          @w
=�G���׿��H��G�CS��G���
=��(��ԏ\CL�                                    Bx_��  �          @s�
�Mp��޸R��ff����CPaH�Mp�����\��
=CJ��                                    Bx_값  �          @z=q�W
=���O\)�?�CPQ��W
=���R�����\)CK�)                                    Bx_�8  �          @|(��[���{����=qCP� �[��У׿��\�pz�CMp�                                    Bx_���  �          @z=q�aG���z�fff�V=qCI�H�aG���=q�����{CE                                    Bx_�܄  �          @|(��^{��G������|z�CK��^{��\)��p���(�CE�)                                    Bx_��*  �          @|���e���p��B�\�0��CJp��e���
=������CFB�                                    Bx_���  �          @~{�g���p���R���CJ5��g���(�����s�CF��                                    Bx_�v  �          @y���a녿˅�u�g
=CL33�a녿�Q�5�)��CJ(�                                    Bx_�  �          @tz��`  �\�#�
���CK���`  ��Q�   ����CJW
                                    Bx_�%�  �          @s�
�`  ���R=�G�?�\)CK��`  ���H��33���CJ�
                                    Bx_�4h  �          @u��aG���p�>�=q@~{CJ�q�aG����R�L���=p�CJ��                                    Bx_�C  T          @w��c�
��Q�>�ff@�z�CJ��c�
��G��#�
�W
=CJ�                                    Bx_�Q�  �          @o\)�Z�H����?+�A$z�CI���Z�H���R>aG�@Z=qCK��                                    Bx_�`Z  �          @p  �\(���  ?G�A@��CG��\(���
=>�Q�@���CJ�                                    Bx_�o   �          @q��`  ��=q?k�AbffCE=q�`  ����?
=qA��CH��                                    Bx_�}�  �          @q��Z�H����?fffA\(�CI�=�Z�H�Ǯ>�G�@�CL��                                    Bx_�L  �          @q��W�����?��A��CIc��W��˅?!G�Az�CME                                    Bx_��  �          @p  �Tzΰ�
?�33A��CI
=�Tz����?333A+�CMO\                                    Bx_멘  �          @mp��P�׿�=q?���A�CFG��P�׿���?z�HAv=qCL�                                    Bx_�>  R          @l���QG����\?�33A��CEJ=�QG���33?�G�A}�CK33                                   Bx_���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_�Պ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_���  �          @i���N{��z�?�p�A�Q�CG�\�N{���R?O\)AL��CL��                                    Bx_�|  
�          @g
=�L�Ϳ�\)?p��AqG�CK��L�Ϳ˅>�@�33CNp�                                    Bx_�"  �          @i���Tzῆff?�=qA��
CE���TzῪ=q?333A0  CI��                                    Bx_��  
�          @i���Tz῝p�?L��AL��CH\)�Tz῵>\@�CK33                                    Bx_�-n  
�          @j�H�U�����>aG�@Z=qCK���U���Q쾀  �\)CKp�                                    Bx_�<  �          @E��.{��Q�u��G�CO�
�.{���
�0���MCMJ=                                    Bx_�J�  �          @C�
�0�׿�ff=�G�@Q�CM@ �0�׿��\���R���CL�R                                    Bx_�Y`  "          @L���9����  >Ǯ@޸RCKG��9����ff���\)CL=q                                    Bx_�h  "          @?\)�%���{?�G�A��C;s3�%��8Q�?�=qA��CC�
                                    Bx_�v�  T          @Q��G��^�R?�Ap�CCz��G��}p�>W
=@p  CE��                                    Bx_�R  �          @Z�H�N�R��  ��\)��p�CE33�N�R�\(��z���CB��                                    Bx_��  
�          @U��AG���{�L�ͿW
=CLE�AG����
���{CJ�H                                    Bx_좞  �          @G��5���>�(�A�CI�)�5��(�=#�
?Tz�CK8R                                    Bx_�D  T          @X���G
=����>\)@�CJ��G
=�����z���z�CJ�\                                    Bx_��  
�          @X���G����\>�\)@���CJ5��G���ff�\)�(�CJ�\                                    Bx_�ΐ  
�          @Z�H�C33��(�>��HA�CM���C33����#�
�8Q�CN��                                    Bx_��6  "          @P���=p���G�>�G�@��\CK��=p����<#�
>B�\CL@                                     Bx_���  T          @P  �B�\���?   AG�CF���B�\���>\)@��CH�
                                    Bx_���  �          @^�R�K���G�?
=qA
=CI���K���\)=���?У�CKL�                                    Bx_�	(  �          @`  �L�Ϳ��?n{Aw�CFE�L�Ϳ�ff?
=qACJ&f                                    Bx_��  �          @Y���E���\)?fffAup�CG��E�����>��HA�CK�f                                    Bx_�&t  
�          @^�R�N{��(�>Ǯ@θRCH�\�N{���
����CI�q                                    Bx_�5  �          @P  �<(�����>�G�@�
=CJ��<(����\<�?�\CKp�                                    Bx_�C�  �          @O\)�@  ��\?���A�33C=���@  �Y��?^�RAz=qCC                                    Bx_�Rf  
�          @Vff�H�ÿ
=q?�ffA�  C=ٚ�H�ÿ\(�?O\)A`z�CCQ�                                    Bx_�a  
�          @Z=q�P  �#�
?J=qAU��C?{�P  �\(�?�A  CB�\                                    Bx_�o�  T          @Vff�K��5?B�\AQ��C@�
�K��k�>�AQ�CD&f                                    Bx_�~X  "          @U��J�H�J=q?�RA,  CA�R�J�H�s33>��
@��CD��                                    Bx_��  "          @W��L�ͿY��?��A$z�CB�H�L�Ϳ�  >�z�@��CE^�                                    Bx_훤  �          @XQ��I����33>���@�33CH�I����
=���Ϳ�
=CH��                                    Bx_��J  
�          @\���L(���G�>�=q@�  CI�{�L(����
�#�
�0  CI�)                                    Bx_���  T          @j�H�Z=q����>.{@.{CI!H�Z=q��ff��\)���CH�H                                    Bx_�ǖ  �          @u��j�H��G�>��@߮CCs3�j�H��{=���?\CD�)                                    Bx_��<  
�          @hQ��^{�O\)?
=Ap�CA��^{�u>�\)@�  CC^�                                    Bx_���  
Z          @c33�Y���&ff?B�\AG
=C>�{�Y���^�R?   A=qCBT{                                    Bx_��  �          @p���e��W
=?=p�A5�CA:��e����>��@��HCD8R                                    Bx_�.  T          @r�\�g��n{?!G�ACBh��g����>�\)@���CD�q                                    Bx_��  "          @{��n�R���\?+�A=qCCaH�n�R��Q�>�z�@�\)CE�q                                    Bx_�z  T          @|���q녿aG�?333A#�
CA��q녿��>�Q�@���CC��                                    Bx_�.   �          @~�R�s�
�k�?.{A=qCA���s�
���>���@�G�CD�                                    Bx_�<�  �          @|(��u�!G�?!G�A\)C=@ �u�L��>\@��C?��                                    Bx_�Kl  
�          @}p��u�J=q?�\@�p�C?�)�u�k�>W
=@ECAff                                    Bx_�Z  �          @~{�s33�s33?&ffACB��s33��{>�\)@�33CDT{                                    Bx_�h�  
[          @x���p  �z�H>���@��CB�{�p  ���
���
����CCY�                                    Bx_�w^  �          @\)�x�ÿTz�>���@�C@��x�ÿc�
<��
>��RC@��                                    Bx_�  "          @\)�y���J=q>�z�@�ffC?n�y���W
=<#�
=�\)C@(�                                    Bx_  
�          @\)�y���E�>�  @h��C?:��y���O\)���ǮC?�=                                    Bx_�P  �          @p���l(���R>�33@�C=� �l(��333=�?�{C>�                                    Bx_��  
Z          @u��q녿�����
��p�C=��q녿
=q��\)��Q�C<)                                    Bx_���  
�          @q��n�R���L���EC;޸�n�R��녾�p����C:W
                                    Bx_��B  �          @j�H�e�.{�u�p��C>�e�\)����
=C<�{                                    Bx_���  "          @fff�aG��#�
��33��G�C>E�aG��������C;�                                     Bx_��  �          @e�`�׿�R�\�ÅC=�3�`�׾�ff����\C;=q                                    Bx_��4  
�          @hQ��Z�H��=q����CE}q�Z�H�Y���O\)�N�\CA�R                                    Bx_�	�  
�          @g��XQ쿑녿z��(�CF���XQ�^�R�p���qp�CBaH                                    Bx_��  �          @n{�^�R��G���33���
CG�f�^�R����B�\�=CD�                                    Bx_�'&  T          @b�\�]p��z��ff��(�C=s3�]p��\�!G��#\)C:L�                                    Bx_�5�  T          @c33�Z�H�:�H�����HC@��Z�H���O\)�T  C;��                                    Bx_�Dr  
�          @g
=�Y���&ff�}p���C>�)�Y�������z����
C8u�                                    Bx_�S  
�          @g��R�\������\)C;�f�R�\=�\)���R��
=C2�=                                    Bx_�a�  �          @j=q�B�\�:�H�������RCA�=�B�\�#�
���R��C4��                                    Bx_�pd  �          @j=q�P�׿.{��  ���\C?�
�P�׾L�Ϳ�z�����C7��                                    Bx_�
  
�          @hQ��QG��   ��33��
=C<���QG�<��
��p���
=C3�{                                    Bx_  "          @c�
�Vff��Ϳ��
��
=C=T{�Vff�����z���ffC6�)                                    Bx_�V  
�          @g��Z=q�h�ÿ(���)p�CC  �Z=q��R�p���r�RC>W
                                    Bx_��  �          @qG��g
=�k��\(��V�HC7�3�g
==�Q�c�
�]C2��                                    Bx_﹢  �          @mp��g��#�
�O\)�I�C4��g�>�\)�B�\�<z�C/�{                                    Bx_��H  
�          @tz��n�R>#�
�Q��FffC1��n�R>�G��5�+\)C-=q                                    Bx_���  "          @u�p  =�G��L���Ap�C2L��p  >Ǯ�5�+
=C.�                                    Bx_��  �          @x���q�=��aG��R�HC2.�q�>�(��G��:�\C-�\                                    Bx_��:  T          @l���g
=���ÿ0���,(�C9(��g
=�u�B�\�>{C4�                                    Bx_��  �          @i���\�Ϳ@  �@  �@  C@Q��\�;�G��xQ��yp�C;5�                                    Bx_��  �          @l(��aG���G����
���C;0��aG��L�Ϳ�\)���HC4Ǯ                                    Bx_� ,  "          @q��fff��
=��ff��G�C:���fff���
�������C4E                                    Bx_�.�  
�          @|(��s�
���
�u�`��C8���s�
=#�
��G��m�C3\)                                    Bx_�=x  
�          @q��i���L�Ϳh���_�
C7+��i��>\)�k��b�HC1�{                                    Bx_�L  T          @n�R�`  ����(����C6!H�`  >��
��
=����C.�)                                    Bx_�Z�  
Z          @n{�]p�>�����\)C1�
�]p�?
=q��ff���HC+{                                    Bx_�ij  "          @l(��Fff?�\)������=qC .�Fff?���c�
�f=qC��                                    Bx_�x  	`          @p���\��>Ǯ��Q����\C-���\��?aG����H���C%��                                    Bx_���  �          @e�P  ?�Ϳ����z�C*h��P  ?�G�������\C"�                                    Bx_�\  �          @a��I��?Q녿�\)��=qC%h��I��?�G��z�H���\C5�                                    Bx_�  �          @hQ��N{?W
=��(����HC%ff�N{?��������z�CǮ                                    Bx_�  �          @k��X��>\)�������C1���X��?(������C)�
                                    Bx_��N  �          @|(��p  ���Ϳ�
=��\)C5}q�p  >�{�������
C.޸                                    Bx_���  �          @{��n�R=�Q쿞�R��\)C2�f�n�R?���\)��
=C+ٚ                                    Bx_�ޚ  T          @y���k�<#�
�����p�C3��k�>��H��(����C,\)                                    Bx_��@  �          @s�
�h��?W
=�8Q��/�C'�h��?�ff��p�����C#�R                                    Bx_���  T          @tz��j=q?�G���  �l(�C$���j=q?��
>\)@��C$B�                                    Bx_�
�  �          @}p��qG�?#�
�n{�Z�HC*c��qG�?p�׿!G���
C&�                                    Bx_�2  �          @x���j=q>�
=��G���33C-k��j=q?Y�����\�r�\C&�
                                    Bx_�'�  �          @r�\�k�<��c�
�Z{C3�
�k�>�33�Q��H(�C.��                                    Bx_�6~  �          @tz��l��>aG��fff�Z{C0���l��?��B�\�6�HC+�\                                    Bx_�E$  �          @x���qG�=L�Ϳfff�VffC3O\�qG�>�p��Q��C
=C.Y�                                    Bx_�S�  �          @z=q�xQ�k�������C7^��xQ�u����Q�C4�                                    Bx_�bp  �          @|(��xQ쾮{�\)�33C8�q�xQ��G��&ff�\)C5�
                                    Bx_�q  �          @����}p��B�\�0���{C6��}p�=��
�5�#
=C2�H                                    Bx_��  �          @\)�w
=���Ϳ}p��ep�C5���w
=>�=q�u�]�C0                                    Bx_�b  �          @\)�w���\)�u�_�C5��w�>�z�k��UG�C/��                                    Bx_�  �          @\)�z�H���333�"�HC5���z�H>\)�333�"�RC2                                      Bx_�  �          @u�n�R���c�
�W33C5�q�n�R>W
=�aG��S\)C0�                                     Bx_�T  �          @j=q�dz�k��B�\�?�C7��dz�=�\)�J=q�G�C2�H                                    Bx_���  T          @fff�aG��   ������C<
=�aG����R���G�C9                                    Bx_�נ  �          @h���c33���Ϳ
=�ffC:c��c33����333�1��C6c�                                    Bx_��F  �          @e��\�;��
�h���l  C9:��\��=L�Ϳu�yC333                                    Bx_���  �          @l���e��Q�Y���T(�C9� �e���
�k��f�HC4B�                                    Bx_��  �          @mp��g
=��(��0���,(�C:���g
=�\)�L���H(�C68R                                    Bx_�8  �          @s�
�l�;�׿@  �5p�C;@ �l�;#�
�^�R�S�C6h�                                    Bx_� �  �          @s�
�l�Ϳ��=p��2{C<��l�;W
=�aG��Up�C7@                                     Bx_�/�  �          @k��c�
���B�\�>�\C;L��c�
�\)�^�R�\(�C65�                                    Bx_�>*  �          @c�
�\�Ϳz�(��ffC=�)�\�;��R�J=q�Lz�C9&f                                    Bx_�L�  �          @dz��^{�k��L���P(�C7޸�^{=�Q�Tz��W�
C2�{                                    Bx_�[v  �          @g
=�`�׾�\)�J=q�J�RC8���`��=#�
�W
=�W
=C3Y�                                    Bx_�j  �          @e��`  �B�\�.{�0��C7��`  =��
�5�6ffC2�H                                    Bx_�x�  �          @dz��`  �L�Ϳ.{�/
=C7E�`  =�\)�333�5��C2��                                    Bx_�h  T          @mp��j�H�8Q���   C6�j�H<��
=q�33C3}q                                    Bx_�  �          @g��e����������ʏ\C8���e�������{C6
=                                    Bx_�  �          @[��L(����H��=q���CH�{�L(����\�5�@Q�CE��                                    Bx_�Z  �          @`  �U�p�׾�{���\CC�q�U�=p��+��1C@��                                    Bx_��   �          @e�Z=q���\�\��G�CD���Z=q�L�Ϳ=p��=��CA�                                    Bx_�Ц  �          @Y���S�
�+����
��=qC?���S�
�
=������33C>(�                                    Bx_��L  T          @X���P�׿J=q������\)CA�)�P�׿z�+��6ffC>�                                    Bx_���  �          @[��U��:�H��\)���C@ff�U��녿
=q�z�C=�R                                    Bx_���  �          @[��Tz�&ff��\�	�C?
=�Tz���Ϳ8Q��B�HC:��                                    Bx_�>  �          @Vff�O\)�+�����C?�)�O\)��녿@  �MC;:�                                    Bx_��  �          @U�QG����H��
=��=qC<��QG���z�z�� z�C9�                                    Bx_�(�  �          @P  �L(����H�u��\)C<�L(���Q������C:xR                                    Bx_�70  �          @N{�K����#�
�#�
C<L��K���(��B�\�U�C;�)                                    Bx_�E�  �          @N{�Fff�(���
=���C?&f�Fff���Ϳ�R�3�C;h�                                    Bx_�T|  �          @O\)�I���
=��{���HC>���I����녿
=q��HC;ff                                    Bx_�c"  �          @G��E����=�Q�?�=qC;�H�E���녽��
��p�C;�f                                    Bx_�q�  �          @HQ��Dz����#�
�.{C?&f�Dz��;u��ffC>=q                                    Bx_�n  �          @G��C33�!G��#�
�W
=C?���C33�zᾅ���ffC>                                    Bx_�  �          @C�
�?\)��=���?�Q�C>���?\)�녾��=qC>�=                                    Bx_�  �          @A��9���G����R��G�CB���9������z��1C?��                                    Bx_�`  �          @E�@  �.{��z����C@Ǯ�@  �����p�C=                                    Bx_�  
�          @@  �6ff<#�
�Y����G�C3ٚ�6ff>�{�G��t(�C-8R                                    Bx_�ɬ  
�          @?\)�<(����
����RC:L��<(���  �u���C8ٚ                                    Bx_��R  
�          @H���E�W
=���
��C7��E��\)�\�߮C5J=                                    Bx_���  "          @G
=�Dz�Ǯ�W
=�r�\C;(��Dzᾏ\)��{����C9�                                    Bx_���  V          @C�
�@�׾�G��B�\�h��C<T{�@�׾��þ�33��33C:B�                                    Bx_�D  
X          @@  �;����L���w�C>T{�;���녾Ǯ���HC<                                    Bx_��  T          @=p��8�ÿ�\��=q��\)C=�q�8�þ�Q��ff�\)C;+�                                    Bx_�!�  
�          @<(��8Q쾏\)��33���HC9���8Q����(���C6h�                                    Bx_�06  T          @@���=p��\�8Q��b�\C;\)�=p���\)���
��z�C9aH                                    Bx_�>�  
�          @S33�N�R��������G�C<��N�R������� (�C9L�                                    Bx_�M�  �          @U�S33��33�8Q��AG�C9���S33��  ������(�C8Q�                                    Bx_�\(  
�          @Z=q�P�׽�Q�z�H��p�C5��P��>����n{�~=qC.�f                                    Bx_�j�  T          @W
=�I�����������
C6W
�I��>��ÿ�����C.�                                    Bx_�yt  
Z          @Fff�=p������R�:{C?}q�=p����R�O\)�s33C9�                                    Bx_�  �          @@���5��z�H���
��G�CG��5��c�
�����=qCEp�                                    Bx_���  
�          @S33�G�=�Q�p�����\C2ff�G�>�ff�Tz��l��C+��                                    Bx_��f  �          @Z�H�P�׼#�
���\����C4��P��>��Ϳp���
=C-�                                    Bx_��  �          @aG��X�þaG��c�
�l  C7���X��>���h���pQ�C1�                                     Bx_�²  
(          @c33�X�þk��xQ��
=C7�f�X��>.{�z�H��33C1
                                    Bx_��X  T          @P���E���z�}p���{C9xR�E�=����
��\)C1�R                                    Bx_���  �          @c33�Z=q���ÿh���n�HC9u��Z=q=u�xQ��}p�C2�                                    Bx_��  "          @Mp��Dz���fff��Q�C68R�Dz�>�  �^�R�|��C/c�                                    Bx_��J  
�          @Mp��<(�?0�׿����(�C&���<(�?�ff�.{�Dz�C T{                                    Bx_��  �          @R�\�Dz��G��������C<.�Dz�<��
��33��{C3�                                    Bx_��  T          @P  �AG��   ��ff��G�C=aH�AG��#�
��z���(�C4�q                                    Bx_�)<  
Z          @.�R�ff?s33����� �
C���ff?����s33���Cc�                                    Bx_�7�  
�          @1��{>k�������(�C.���{?0�׿�G���G�C$c�                                    Bx_�F�  
�          @6ff�*=q>.{�z�H���C0aH�*=q?�ͿTz���C(k�                                    Bx_�U.  
�          @0  �!G�?�\�^�R��\)C(�H�!G�?O\)�
=�J{C"#�                                    Bx_�c�  �          @1�>\@33��\)���HB�W
>\@p�?(��A�B���                                    Bx_�rz  "          @Fff?�
=@.�R>�@{By�?�
=@��?�(�A��
Bn��                                    Bx_��   
�          @A�?���@*=q>�{@�Bz{?���@33?���A�G�Bk�                                    Bx_���  �          @I��?��@.�R=�@  Br�?��@��?��HA�\)Bg=q                                    Bx_��l  "          @?\)?��
@+�=�@33B��)?��
@=q?�Q�A�=qBw                                    Bx_��  �          @>{?��R@#�
=�@�
Bo{?��R@�\?��A���Bc�                                    Bx_���  �          @=p�?���@'
=>#�
@G�Bxz�?���@z�?���A�G�Bm(�                                    Bx_��^  �          @:�H?���@%�>#�
@C33By�?���@33?�
=A�\)BnG�                                    Bx_��  
�          @:=q?�(�@\)>8Q�@eBm�H?�(�@p�?�A��RBa\)                                    Bx_��  �          @%��u@ff���P��B�Ǯ�u@��?Tz�A�p�B��\                                    Bx_��P  
�          @@  �1G�����ff��33C6aH�1G�>��
��  ��=qC-xR                                    Bx_��  �          @>{�3�
=��Ϳp�����C1���3�
>�׿Q���(�C*ff                                    Bx_��  �          @:�H�3�
<��=p��k�C3L��3�
>��ÿ(���R�\C-J=                                    Bx_�"B  T          @7
=�/\)=�\)�L����G�C2� �/\)>Ǯ�333�b=qC+��                                    Bx_�0�  �          @8���1G�>��G��|(�C1\)�1G�>�(��(���U�C+
                                    Bx_�?�  "          @=p��7�<��5�^=qC3n�7�>��R�#�
�G
=C-                                    Bx_�N4  "          @<���/\)>�(���  ����C+��/\)?L�Ϳ=p��h  C#�3                                    Bx_�\�  
�          @>{�-p�?���{����C)��-p�?n{�L���yG�C!{                                    Bx_�k�  "          @2�\�(�?.{��33��\)C$��(�?���G����C                                    Bx_�z&  "          @2�\���?s33���
��CB����?�녿G����HC�                                    Bx_���  �          @#�
����@	��?0��A33B�\)����?�
=?�G�B�HB��R                                    Bx_��r  "          @����H?��
���\(�CJ=���H?��?333A��\C��                                    Bx_��  �          @�
�\?˅�(����G�C��\?�(�=L��?�C��                                    Bx_���  �          @G�����?�{�k���  Cp�����?Ǯ>�ffA7\)CL�                                    Bx_��d  
�          @z��ff?�(����Q�C���ff?˅?+�A��Cff                                    Bx_��
  
�          @���
?�G��L����=qC!H���
?�
=?
=qAU�CW
                                    Bx_��  �          @�
���H?�G���\)�߮C�)���H?�(�>�A:ffC@                                     Bx_��V  T          @���?�׾���j�HB�  ���?�\?!G�Az�HC �                                    Bx_���  "          @����?�(��L�Ϳ�G�B�ff���?���?B�\A�B��3                                    Bx_��  
�          @�H���
@�\�aG����\B�����
?���?!G�Ar�\B�ff                                    Bx_�H  "          @=q���@ �׾B�\��33B�� ���?�z�?&ffA{�B�#�                                    Bx_�)�  
�          @Q쿧�?�Q�\�  B�=q���?�
=>�ffA.ffB���                                    Bx_�8�  
�          @
=���?�33�����p�B�33���?�33>�
=A#�
B�L�                                    Bx_�G:  �          @녿��?�G���z���=qC	=q���?�  >�Q�AQ�C	�=                                    Bx_�U�  T          @���(�?�33�aG�����B�\)��(�@���
�
=qB�aH                                    Bx_�d�  �          @
=���\?��H�=p����B�����\@>�@U�B�#�                                    Bx_�s,  
�          @ff��{?�׽��
���RB��\��{?�  ?333A��HB���                                    Bx_���  
�          @z�Ǯ?ٙ��#�
��Q�C�\�Ǯ?�{?��A]��C                                    Bx_��x  �          @�\���R?ٙ�����θRC8R���R?�33>��A>ffC                                      Bx_��  
�          @�׿��?˅���
�G�C&f���?�=q>�Q�Az�CJ=                                    Bx_���  �          @�Ϳ˅?��R��  ����Cٚ�˅?��H>\A!��C	ff                                    Bx_��j  
(          @	�����
?��R�B�\��p�C���
?�Q�>�G�A=��C��                                    Bx_��  "          @�Ϳ�z�?�=q��Q���RC����z�?�p�?�AzffC�\                                    Bx_�ٶ  �          @�
��?Tz᾽p���C����?h��<#�
>���C�                                    Bx_��\  
�          @��	��?Tzᾊ=q�׮C��	��?^�R=���@
=C
=                                    Bx_��  
Z          @�(�?G��u���HC \)�(�?O\)=�G�@(��C��                                    Bx_��  
�          @�\�Q�?G������{C�f�Q�?Q�=�Q�@G�C�                                    Bx_�N  "          @\)� ��?xQ���N{Cc�� ��?n{>�z�@�C5�                                    Bx_�"�  
�          @
�H�   ?Q녽��
��C�f�   ?G�>�=q@�RC�3                                    Bx_�1�  �          @G��G�?�  �#�
��=qC�3�G�?z�H>�=q@��HC=q                                    Bx_�@@  "          @(��G�?Q녾#�
��G�C�q�G�?O\)>B�\@���C#�                                    Bx_�N�  
�          @	����\)?��
��Q���HC+���\)?z�H>�33AG�C�                                     Bx_�]�  �          @Q��\?�=u?���Ch���\?��?
=qAl��Cs3                                    Bx_�l2  
�          @G���\?�>�@P��CG���\?�p�?5A�\)C�                                    Bx_�z�  �          @���G�?\<�?L��C5ÿ�G�?�\)?(��A�z�C+�                                    Bx_��~  �          @
=��G�?��ý��
����C
Y���G�?���?z�Aep�Cn                                    Bx_��$  "          @�\��33?���    �L��C����33?�
=?&ffA�33C8R                                    Bx_���  �          @8�ÿ�z�@ff�\)�=�C p���z�@��>�Q�@���B��q                                    Bx_��p  T          @7����>��?�G�A���C*@ ������?��B�\C:)                                    Bx_��  
�          @9���"�\?333?�
=A���C$���"�\>\)?�\)A�{C0޸                                    Bx_�Ҽ  �          @:=q�(��>�G�?��A��HC*n�(�ý�\)?�(�A��HC5�
                                    Bx_��b  
�          @7��)��>#�
?�=qA��C0���)�����R?�ffA�Q�C:��                                    Bx_��  "          @,(��{���?��A�
=C7^��{��?c�
A�ffC@��                                    Bx_���  
�          @(���   ���?Q�A�p�C7� �   ��?.{Ap(�C>�3                                    Bx_�T  �          @"�\�p��#�
?&ffAl��C4E�p���\)?z�AT��C:�\                                    Bx_��  �          @+��%����
?5Aup�C4�=�%����
?!G�AZ�RC;�                                    Bx_�*�  T          @+��$z�>��H?��A=p�C)B��$z�>W
=?333As�C/E                                    Bx_�9F  T          @,(��'
=>�
=>��A�
C*��'
=>W
=?��A<��C/L�                                    Bx_�G�  
�          @1G��/\)>��ý#�
�G�C-.�/\)>��R=���@��C-z�                                    Bx_�V�  T          @,(��*=q>��þ.{�h��C,���*=q>�p����
����C,�                                    Bx_�e8  
�          @0  �)��?&ff�8Q��n{C&+��)��?+�=�G�@�C%�H                                    Bx_�s�  �          @'
=���?c�
<�?�C 
���?J=q>���A  C"
=                                    Bx_���  
�          @"�\�33?��=�Q�@33CW
�33?k�?�A?�
C33                                    Bx_��*  �          @!G��\)?�33=�\)?�=qCٚ�\)?�G�?��AH��C��                                    Bx_���  T          @)����?���#�
�\(�C{��?�G�>��A�HC!H                                    Bx_��v  �          @\)��(�?�?��B	
=C%5ÿ�(���G�?�p�BQ�C7E                                    Bx_��  "          @�ÿ�p�>�?�G�A��C&�ÿ�p���?��B\)C7h�                                    Bx_���  "          @��33>��
?��A�
=C+��33�#�
?���A�
=C8aH                                    Bx_��h  �          @�
��\>���?��
Aә�C*� ��\��?���A�p�C7��                                    Bx_��  �          @�\��>�
=?W
=A�\)C(����<�?p��A���C3G�                                    Bx_���  
�          @��G�>��?��Ak33C-k��G����
?&ffA��\C4c�                                    Bx_�Z  �          @����Q�?:�H?�Q�A��CG���Q�>#�
?��Bz�C/J=                                    Bx_�   
�          @ff�
�H?�R?�RAv�HC$!H�
�H>���?Q�A��C,8R                                    Bx_�#�  
Z          @=q�33?(�>�\)@׮C%.�33>�(�?�\AEp�C)W
                                    Bx_�2L  �          @33�\)?�\�#�
�B�\C'{�\)>��>W
=@�  C(.                                    Bx_�@�  T          @z��\)?\)�#�
�#�
C&��\)?�\>k�@�C'J=                                    Bx_�O�  T          @ff��?�ͽ��
��(�C&z���?�>#�
@xQ�C&��                                    Bx_�^>  T          @\)��H?
=q�.{�s33C'����H?\)=��
?��C'�                                    Bx_�l�  "          @%� ��?
=��  ����C&�3� ��?#�
<��
?�C%�)                                    Bx_�{�  "          @%��   ?녾��R���HC'��   ?&ff�#�
�L��C%p�                                    Bx_��0  �          @ �����>��H���D��C(�����?+���\)���HC$xR                                    Bx_���  �          @,���p�>��z�H��33C(�H�p�?\(��0���k�
C ٚ                                    Bx_��|  �          @(Q���R?��+��mp�C(���R?B�\�Ǯ��
C#                                      Bx_��"  �          @333�*�H?���z��?
=C'J=�*�H?J=q��=q����C#�                                     Bx_���  "          @0���'�?\)�!G��T��C'���'�?G���{����C#��                                    Bx_��n  �          @-p���R?�\�n{���RC(z���R?Y����R�V{C!�                                    Bx_��  T          @-p��{>���  ���HC)
=�{?\(��333�o�C ��                                    Bx_��  #          @+��{?   �k����C(� �{?Y����R�V�HC!�                                    Bx_��`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_��  �          @"�\�Q�>�녿G�����C*:��Q�?333���@��C#�
                                    Bx_�+R  �          @#33�Q�>�p��Q����HC++��Q�?.{�
=�U��C#��                                    Bx_�9�  	�          @����>�G��E����C(�R���?8Q��\�A�C"33                                    Bx_�H�  
�          @���>\�.{���C*G����?!G����4Q�C$�                                    Bx_�WD  
]          @�\�
=>Ǯ�G����C)� �
=?0�׿
=q�Z�\C"                                      Bx_�e�  Q          @�
�	��>��Ϳ=p����
C)z��	��?.{�   �H��C"�                                    Bx_�t�  %          @���(�>Ǯ�aG���\)C)���(�?:�H�!G��u�C!��                                    Bx_��6  �          @#�
��>��n{���HC(p���?Tz�#�
�g�
C u�                                    Bx_���  
�          @ ����>��n{��(�C(���?Tz�#�
�k�C��                                    Bx_���  T          @�
�G�>��
��ff��=qC*���G�?=p��O\)��z�C�3                                    Bx_��(  
Z          @����
=�Q����#Q�C1B����
?:�H��{�C�H                                    Bx_���  
�          @33���>�{��33�ffC)@ ���?fff��\)��Q�C0�                                    Bx_��t  �          @�
��
=?^�R=���@��C�3��
=?=p�>�A��C��                                    Bx_��  "          @�׿�Q�?�33>�p�A=qC:ῸQ�?�=q?��A��HC	L�                                    Bx_���  
�          @��?�ff>�A5C E��?�
=?�
=A�33C                                    Bx_��f  T          @"�\�У�?��>W
=@�p�C��У�?�{?}p�A���Cs3                                    Bx_�  �          @,(���\?�z���7�C޸��\?�p�>k�@���C�R                                    Bx_��  T          @"�\����?���
=q�C\)C�R����?�\)>.{@u�CB�                                    Bx_�$X  �          @'
=����?�z���H�-�C�Ϳ���?��H>�=q@��HC
�3                                    Bx_�2�  T          @2�\�?�G�����6�RC�H�?���>�  @�p�C
��                                    Bx_�A�  �          @0����
?��
��ff��C)��
?�ff>�33@�Q�C
�q                                    Bx_�PJ  T          @(Q��
=?�G���{���C	�f��
=?�p�>�ffAQ�C
\                                    Bx_�^�  T          @%����?޸R��\)�ə�C	B����?�Q�>��HA-C
                                    Bx_�m�  "          @,�Ϳ�Q�?�{��z���p�C5ÿ�Q�?�ff?�A5�C	�                                    Bx_�|<  T          @.�R����?�׾�z��\C&f����?���?
=qA6�RC	                                    Bx_���  "          @(Q���H?�(���33���HC
�ÿ��H?ٙ�>�
=A�HC
��                                    Bx_���  
�          @.{��
?�(���ff��HC33��
?�  >���@޸RCǮ                                    Bx_��.  �          @0���	��?��R�^�R��\)C33�	��?�(�����H��CG�                                    Bx_���  T          @,����?���h������CxR��?�ff��=q��G�C��                                    Bx_��z  
�          @(����?��׿�{��  C�)��?�  ���4��C�                                     Bx_��   
�          @\)�   ?W
=��
=���HCQ��   ?�G��5��=qCǮ                                    Bx_���  "          @�H�
=��G����\�Σ�C6���
=>�{�xQ���\)C*ٚ                                    Bx_��l  �          @
=�\)��
=���Q�C>���\)����(�����C7�                                    Bx_�   	�          @��{=�>�{A  C0�=�{��>�Q�Az�C4�)                                    Bx_��  �          @�
��G�?p��?���Bp�C��G�>��?�33B<��C*5�                                    Bx_�^  
�          @'����R@
==�\)?�{B�𤿞�R?��?uA��B��                                    Bx_�,  �          @,(��xQ�@p�<#�
>�z�B�aH�xQ�?��R?uA�Q�B��H                                    Bx_�:�  "          @1녿!G�@*�H=�G�@G�BΔ{�!G�@
=?��RA�ffB���                                    Bx_�IP  "          @1녿G�@&ff����F�RB�G��G�@&ff?�A=�B�33                                    Bx_�W�  T          @,�ͿTz�@=q?L��A�B�=q�Tz�?�=q?�  B$
=B��                                    Bx_�f�  T          @.{�333@"�\?&ffA\��BҨ��333@G�?�B�HB�                                      Bx_�uB  "          @1G���\@ ��?���A�33B����\?�@�B<��B�k�                                    Bx_���  T          @,(���  ?�
=>u@�ffB��ῠ  ?�33?��A��
B���                                    Bx_���  "          @9����
=@�\�Y����p�B�R��
=@�>W
=@�
=B���                                    Bx_��4  �          @H�ÿ��\@333?h��A��B�녿��\@��@ ��B!Q�B�q                                    Bx_���  "          @C�
����@
=��(��33C \����@z�?�RA;�
C ��                                    Bx_���  "          @J=q�ff@�
������(�CY��ff@��?�RA5p�C�3                                    Bx_��&  "          @U��R@
=�@  �P��CaH��R@��>���@��
C@                                     Bx_���  �          @[��  @������33C��  @%�<#�
=�G�C)                                    Bx_��r  "          @Y�����@
=�}p���ffC����@#�
=�G�?�C��                                    Bx_��  "          @U��H?�p���G��י�Cu���H@p��!G��.=qC	�)                                    Bx_��  
�          @Q��{?\�\��p�CT{�{@녿:�H�M�C��                                    Bx_�d  
�          @QG����?��Ϳ�
=����C{���?�Q�p����  C��                                    Bx_�%
  �          @L����\?��\���
\)C�q��\?��������C
=                                    Bx_�3�  W          @C33���?5���A�Cs3���?�{��G��
=C�\                                    Bx_�BV            @;���?�33����#�C�ÿ�?����H��p�C�                                    Bx_�P�  T          @�H�Ǯ?���33�  C0��Ǯ?�33�E����CQ�                                    Bx_�_�  �          @W�@"�\���
?���A���C���@"�\��=q?^�RAyp�C�"�                                    Bx_�nH  
(          @u@B�\���
?�A��C���@B�\��(�?�Q�A�33C���                                    Bx_�|�  
�          @e@7
=����?�ffA�C�w
@7
=��\?�z�A�Q�C�*=                                    Bx_���  �          @xQ�@G
=���\?�Q�AC�޸@G
=��p�?��HA�\)C�Ǯ                                    Bx_��:  "          @u@G
=���H?�z�A�z�C�\)@G
=��z�?��HA�33C�8R                                    Bx_���  T          @j=q@?\)���?���A�RC�q�@?\)��p�?��RA�\)C�H                                    Bx_���  �          @Vff@333�#�
?�p�A�=qC��
@333���?��A�\)C�q�                                    Bx_��,  
�          @\��@:�H�.{?�(�A�Q�C�z�@:�H���?�G�A��
C���                                    Bx_���  
�          @^{@>{�!G�?�
=A�{C���@>{���?�  A�G�C�>�                                    Bx_��x  �          @b�\@AG����?�  A��C�\)@AG����?�=qA�=qC�n                                    Bx_��  "          @aG�@;��.{?���A��RC�xR@;����?���A��C�U�                                    Bx_� �  
�          @c�
@?\)�5?�ffA�{C�L�@?\)��z�?���A�(�C�e                                    Bx_�j  �          @vff@N�R�5?��HA�
=C��@N�R��(�?�(�A��C���                                    Bx_�  �          @�Q�@Z=q�#�
@ ��A�Q�C���@Z=q��Q�?��A���C���                                    Bx_�,�  "          @vff@S�
��G�?�A��C�.@S�
��(�?�ffA���C��                                    Bx_�;\  �          @s�
@Q녾���?��A�ffC��@Q녿�?��A�33C�4{                                    Bx_�J  
�          @w
=@R�\���H?��HA��C���@R�\���\?ǮA�\)C�k�                                    Bx_�X�  
�          @tz�@Tz�\?�=qA�C���@Tz῏\)?�  A���C���                                    Bx_�gN  "          @vff@X�þ.{?���A�  C���@X�ÿn{?�=qA�G�C�P�                                    Bx_�u�  �          @z=q@Z�H�W
=?�\)A�
=C�@ @Z�H�z�H?�{A�=qC���                                    Bx_���  �          @~�R@c33�L��?��
A���C�]q@c33�p��?��
A�z�C��
                                    